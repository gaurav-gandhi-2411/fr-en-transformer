from __future__ import annotations

# Colab-side evaluation steps for `CONFIG = "eval_l4"` (RUNBOOK §4.5). The notebook runs each
# subcommand below as its own subprocess (the kernel imports only the stdlib and torch), in this
# order: hf-verify (the notebook's skip check), hf-check, candidates (final only) + bench,
# candidates, tune (per candidate), select, decode, validate-test, upload. The throughput bench
# runs before any tuning/selection/decoding. Selection happens HERE, on Colab, on E1 + E2 only
# (dev and E3 are only ever DECODED, after the winner is fixed; PREREG §1-§2,
# via nmt.tune / nmt.selection); the local pipeline (scripts/eval_local.py) never decodes or
# selects, it only scores what this module uploaded.
#
# Every step is idempotent: its output is written last, and a re-run skips it when the output
# exists and validates (a Colab disconnect resumes where it stopped). A step that cannot verify
# something fails closed (nonzero exit), never defaults to the permissive branch.
#
# Directory layout under <eval_dir> (Drive: MyDrive/fr-en-transformer/eval/<RUN>):
#   candidates/<name>/{model.safetensors,config.json,spm.model,candidate_meta.json}
#   bench.json  tuning/<name>.json  selection.json  decode_summary.json  validation.json
#   predictions/{seg_off,seg_tuned}/<split>_predictions.json   test_predictions.json (main)
#   run_meta.json  hf_upload.json
#
# CLI: `python -m nmt.eval_l4 <subcommand> ...` (see `--help` per subcommand).
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNS: tuple[str, ...] = ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
DECODE_SPLITS: tuple[str, ...] = ("dev", "e1", "e2", "e2synth", "e3")
VARIANT_SEG_OFF = "seg_off"
VARIANT_TUNED = "seg_tuned"
VARIANTS: tuple[str, ...] = (VARIANT_SEG_OFF, VARIANT_TUNED)
# Decode batch size for bench, tuning and the final decodes. 32 is Translator.translate's own
# default; one value everywhere so the bench rate predicts the tuning/decoding time.
EVAL_BATCH_SIZE = 32
BENCH_N_SENTENCES = 200
BENCH_ALPHA = 0.6
BENCH_BEAM = 5
# Candidates whose objectives differ by <= this are tied; the tie goes to the earliest candidate
# in the order passed to `select` (final first), i.e. the one with the fewest averaged checkpoints.
TIE_EPSILON = 1e-9
TIE_RULE = (
    f"candidates within {TIE_EPSILON:g} of the best objective are tied; the tie goes to the "
    "earliest candidate in the pre-registered order (final, avg_last5, avg_decay)"
)
# The candidates each run's selection compares (PREREG §6, 2026-10-02 (b)); the notebook holds the
# same lists as step numbers and tests/test_colab_eval_l4.py checks the two agree.
MAIN_CANDIDATES: tuple[str, ...] = ("final", "avg_last5", "avg_decay")
ABLATION_CANDIDATES: tuple[str, ...] = ("final",)
MANIFEST_NAME = "manifest.json"
EXPECTED_TEST_IDS = 330  # data/test/sample_submission.json
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


class EvalStepError(RuntimeError):
    """A step refused to continue (missing input, invalid output, bad parameter)."""


class HFWriteTokenError(EvalStepError):
    """The HF token cannot write to the target repo (read-only or wrong scope)."""


class HFNotPrivateError(EvalStepError):
    """The target HF repo is not private (or its privacy could not be read back)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    tmp.write_bytes(text.encode("utf-8"))  # bytes: LF on every OS, so files diff/hash alike
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written "valid" output


def _read_json(path: Path) -> Any | None:
    """Parsed JSON, or None when the file is absent or unreadable (fail closed for validators)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# HF token / repo safety
# ---------------------------------------------------------------------------------------------


def write_scope_problem(whoami: dict[str, Any], repo_id: str) -> str | None:
    """Why the token behind `whoami` (HfApi.whoami()) cannot write to `repo_id`, else None.

    Refuses only when it is sure: role "read", or a fine-grained token whose scoped entries carry
    no write permission for this repo. An unrecognised shape returns None; the upload itself is
    the backstop (a 401/403 there is mapped to the same error, nothing is leaked by trying).
    """
    token = ((whoami.get("auth") or {}).get("accessToken")) or {}
    role = token.get("role")
    if role in ("write", "admin"):
        return None
    if role == "read":
        return "the token has role 'read' (read-only)"
    if role == "fineGrained":
        fine = token.get("fineGrained") or {}
        owner = repo_id.split("/", 1)[0]
        grants = [(fine.get("global") or [])]
        for scope in fine.get("scoped") or []:
            entity = scope.get("entity") or {}
            name = entity.get("name")
            if name == repo_id or (entity.get("type") in ("user", "org") and name == owner):
                grants.append(scope.get("permissions") or [])
        if any("repo.write" in perms for perms in grants):
            return None
        return f"the fine-grained token has no write permission ('repo.write') on {repo_id}"
    return None


def write_token_message(repo_id: str, why: str) -> str:
    """The actionable text for a token that cannot write."""
    return (
        f"HF token cannot write to {repo_id}: {why}. GG: create the PRIVATE model repo "
        f"{repo_id} on huggingface.co (New model, visibility Private), then create a "
        "fine-grained token at https://huggingface.co/settings/tokens with 'Write access to "
        f"contents/settings of selected repos' for exactly {repo_id}, store it as the Colab "
        "secret HF_TOKEN_WRITE (Notebook access on), and Run all again."
    )


def _status(exc: BaseException) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _is_auth_error(exc: BaseException) -> bool:
    return _status(exc) in (401, 403)


def check_write_token(api: Any, repo_id: str) -> str:
    """Raise HFWriteTokenError unless the token can plausibly write to `repo_id`; returns the
    account name. Called before any GPU time is spent and again before the upload."""
    try:
        who = api.whoami()
    except Exception as exc:
        raise HFWriteTokenError(
            write_token_message(repo_id, f"whoami failed ({type(exc).__name__}: {exc})")
        ) from exc
    problem = write_scope_problem(who, repo_id)
    if problem:
        raise HFWriteTokenError(write_token_message(repo_id, problem))
    return str(who.get("name", "?"))


def assert_repo_private(api: Any, repo_id: str, when: str) -> None:
    """Read the repo back and raise HFNotPrivateError unless `private is True` (fail closed: a
    missing/None flag is a refusal, not a pass)."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        raise HFNotPrivateError(
            f"{when}: could not read back {repo_id} to verify it is private "
            f"({type(exc).__name__}: {exc}); refusing."
        ) from exc
    if getattr(info, "private", None) is not True:
        raise HFNotPrivateError(
            f"{when}: HF repo {repo_id} is NOT private "
            f"(private={getattr(info, 'private', None)!r}). Refusing to upload; make it Private "
            "on huggingface.co (or use another repo)."
        )


def ensure_private_repo(api: Any, repo_id: str) -> None:
    """create_repo(private=True, exist_ok=True), then read back and refuse unless private."""
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as exc:
        # A repo-scoped fine-grained token may lack "create repos" while the repo already
        # exists; that is fine as long as it can be read back. Anything else is real.
        if not _is_auth_error(exc):
            raise
        print(f"create_repo refused ({exc}); continuing if {repo_id} already exists.")
    assert_repo_private(api, repo_id, "before upload")


PROBE_PATH = "_write_probe.txt"  # repo root, outside runs/: no manifest, verify or pull reads it
PROBE_CONTENT = b"write probe: fr-en-transformer eval repo; safe to ignore\n"
# A title distinct from commit_message(run): hf-verify matches upload commits by exact title.
PROBE_COMMIT_MESSAGE = "eval_l4: write probe"


def _write_probe(api: Any, repo_id: str) -> str:
    """Prove the token can really write: commit one tiny file unless it is already there.

    Returns "committed" or "already present". Presence of PROBE_PATH proves an earlier successful
    write by the repo owner (only a write-capable token can have put it there), so it is not
    repeated; it does not prove THIS token can write, so the upload step's own 401/403 mapping
    remains the backstop for that. 401/403 on the probe is HFWriteTokenError.
    """
    from huggingface_hub import CommitOperationAdd

    try:
        if api.file_exists(repo_id=repo_id, filename=PROBE_PATH, repo_type="model"):
            return "already present"
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[CommitOperationAdd(path_in_repo=PROBE_PATH, path_or_fileobj=PROBE_CONTENT)],
            commit_message=PROBE_COMMIT_MESSAGE,
        )
    except Exception as exc:
        if _is_auth_error(exc):
            raise HFWriteTokenError(
                write_token_message(repo_id, f"write probe refused: {exc}")
            ) from exc
        raise
    return "committed"


def hf_check(api: Any, repo_id: str) -> dict[str, Any]:
    """Early fail-fast check, before any GPU time: token scope, the repo exists and is private
    (created private if missing), and a REAL write probe (token metadata alone cannot prove the
    token can write). Privacy is asserted again after the probe commit."""
    if not _REPO_ID_RE.match(repo_id):
        raise EvalStepError(f"HF_EVAL_REPO={repo_id!r} is not '<owner>/<name>'")
    account = check_write_token(api, repo_id)
    exists = True
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        if _is_auth_error(exc) or getattr(getattr(exc, "response", None), "status_code", 0) != 404:
            raise EvalStepError(
                f"cannot read {repo_id} ({type(exc).__name__}: {exc}); "
                + write_token_message(repo_id, "repo not readable with this token")
            ) from exc
        exists = False
    if exists:
        assert_repo_private(api, repo_id, "hf-check")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        if not exists:  # cannot create it and it is not there: the token cannot write
            raise HFWriteTokenError(
                write_token_message(repo_id, f"create_repo refused: {exc}")
            ) from exc
    assert_repo_private(api, repo_id, "hf-check, before the write probe")
    probe = _write_probe(api, repo_id)
    assert_repo_private(api, repo_id, "hf-check, after the write probe")
    return {"account": account, "repo": repo_id, "exists": exists, "probe": probe}


# ---------------------------------------------------------------------------------------------
# candidates (checkpoint averaging + HF-style export)
# ---------------------------------------------------------------------------------------------


def parse_candidate_specs(specs: Sequence[str]) -> dict[str, list[int]]:
    """`["final=24645", "avg=1,2,3"]` -> {"final": [24645], "avg": [1, 2, 3]} (order kept)."""
    out: dict[str, list[int]] = {}
    for spec in specs:
        name, sep, steps = spec.partition("=")
        if not sep or not name or not steps:
            raise EvalStepError(f"bad --candidate {spec!r}; expected name=step,step,...")
        if name in out:
            raise EvalStepError(f"duplicate candidate {name!r}")
        out[name] = [int(s) for s in steps.split(",")]
    return out


def checkpoint_path(ckpt_dir: Path, step: int) -> Path:
    """nmt.train's checkpoint file name for `step`."""
    return Path(ckpt_dir) / f"step_{step:08d}.pt"


def check_candidate_files(ckpt_dir: Path, specs: dict[str, list[int]]) -> None:
    """Raise EvalStepError naming EVERY missing checkpoint file (never substitutes a neighbour)."""
    missing = [
        str(checkpoint_path(ckpt_dir, s))
        for steps in specs.values()
        for s in steps
        if not checkpoint_path(ckpt_dir, s).is_file()
    ]
    if missing:
        raise EvalStepError(
            "missing checkpoint file(s), refusing to continue: " + ", ".join(sorted(set(missing)))
        )


def candidate_is_valid(out_dir: Path, steps: Sequence[int]) -> bool:
    """True iff `out_dir` holds a finished export of exactly `steps` (meta is written last)."""
    out_dir = Path(out_dir)
    meta = _read_json(out_dir / "candidate_meta.json")
    weights = out_dir / "model.safetensors"
    return bool(
        isinstance(meta, dict)
        and meta.get("steps") == list(steps)
        and weights.is_file()
        and weights.stat().st_size == meta.get("model_bytes")
        and (out_dir / "config.json").is_file()
        and (out_dir / "spm.model").is_file()
    )


def build_candidate(
    name: str,
    steps: Sequence[int],
    ckpt_dir: Path,
    config_path: Path,
    out_dir: Path,
    tokenizer_path: Path | None = None,
) -> bool:
    """Average the checkpoints `steps` (a single step is its own weights) and export an HF-style
    model dir. Returns False when a valid export already exists (skipped), True when built."""
    if candidate_is_valid(out_dir, steps):
        print(f"candidate {name}: SKIP (valid export at {out_dir})")
        return False
    from nmt.hub import export_checkpoint
    from nmt.train import _build_model_config, load_config

    check_candidate_files(ckpt_dir, {name: list(steps)})
    paths = [checkpoint_path(ckpt_dir, s) for s in steps]
    meta_path = Path(out_dir) / "candidate_meta.json"
    meta_path.unlink(missing_ok=True)  # a half-rebuilt dir must never look valid
    cfg = load_config(config_path)
    export_checkpoint(
        paths,
        Path(out_dir),
        _build_model_config(cfg),
        tokenizer_path or REPO_ROOT / "tokenizer" / "spm.model",
        average=True,
    )
    weights = Path(out_dir) / "model.safetensors"
    _write_json(
        meta_path,
        {
            "name": name,
            "steps": list(steps),
            "ckpt_files": [p.name for p in paths],
            "ckpt_bytes": [p.stat().st_size for p in paths],
            "config": Path(config_path).name,
            "model_bytes": weights.stat().st_size,
            "model_sha256": _sha256_file(weights),
        },
    )
    print(f"candidate {name}: built from {[p.name for p in paths]} -> {out_dir}")
    return True


# ---------------------------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------------------------


def _sync(device: Any) -> None:
    import torch

    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize(device)


def bench_translator(
    translator: Any,
    texts: Sequence[str],
    batch_size: int = EVAL_BATCH_SIZE,
    warmup: int = 8,
) -> dict[str, Any]:
    """Time greedy and beam 5 (alpha 0.6, no segmentation) over `texts`, one `translate()` call
    per mode exactly as nmt.tune/nmt.evaluate call it (length-sorted batches). Output tokens are
    SentencePiece tokens of the detokenized hypotheses (no EOS), an approximation of the generated
    count. One untimed warm-up call first so CUDA/cuDNN initialisation is not billed to greedy."""
    texts = list(texts)
    translator.translate(texts[:warmup], batch_size=batch_size, beam=1, segment_threshold=None)
    source_tokens = sum(len(translator.sp.encode(t, out_type=int)) for t in texts)
    modes: dict[str, dict[str, Any]] = {}
    for label, beam in (("greedy", 1), (f"beam{BENCH_BEAM}", BENCH_BEAM)):
        _sync(translator.device)
        t0 = time.perf_counter()
        outputs = translator.translate(
            texts, batch_size=batch_size, beam=beam, alpha=BENCH_ALPHA, segment_threshold=None
        )
        _sync(translator.device)
        wall = time.perf_counter() - t0
        out_tokens = sum(len(translator.sp.encode(h, out_type=int)) for h in outputs)
        modes[label] = {
            "beam": beam,
            "alpha": None if beam == 1 else BENCH_ALPHA,
            "wall_seconds": round(wall, 4),
            "sentences_per_second": round(len(texts) / wall, 3) if wall > 0 else None,
            "output_tokens": out_tokens,
            "output_tokens_per_second": round(out_tokens / wall, 2) if wall > 0 else None,
        }
    return {"n_sentences": len(texts), "source_tokens": source_tokens, "modes": modes}


def run_bench(model_dir: Path, out_path: Path, run: str, batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Throughput benchmark on the first 200 E2 sentences (file order); writes bench.json."""
    import torch

    from nmt.evaluate import load_split
    from nmt.translate import Translator

    inputs, _ = load_split("e2")
    rows = inputs[:BENCH_N_SENTENCES]
    translator = Translator.from_pretrained(str(model_dir))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    result = bench_translator(translator, [r["source"] for r in rows], batch_size)
    ids = [r["id"] for r in rows]
    result.update(
        {
            "run": run,
            "split": "e2",
            "first_id": ids[0],
            "last_id": ids[-1],
            "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "batch_size": batch_size,
            "segmentation": "off",
            "model_dir": str(model_dir),
            "device": str(translator.device),
            "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
            # Translator never uses autocast: decoding runs in the weights' own fp32, whatever
            # precision (bf16/fp16) the model was trained in.
            "decode_precision": "fp32 (no autocast in nmt.translate)",
            "torch": torch.__version__,
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if cuda
            else None,
        }
    )
    _write_json(out_path, result)
    return result


def bench_is_valid(path: Path) -> bool:
    data = _read_json(path)
    if not isinstance(data, dict) or data.get("n_sentences") != BENCH_N_SENTENCES:
        return False
    modes = data.get("modes") or {}
    return all(
        isinstance((modes.get(m) or {}).get("output_tokens_per_second"), int | float)
        for m in ("greedy", f"beam{BENCH_BEAM}")
    )


# ---------------------------------------------------------------------------------------------
# tune + select
# ---------------------------------------------------------------------------------------------


def _full_selection_sizes() -> tuple[int, int]:
    from nmt.selection import load_selection_set

    return len(load_selection_set("e1")), len(load_selection_set("e2"))


def tuning_is_full(tuning: Any, sizes: tuple[int, int] | None = None) -> bool:
    """True iff `tuning` is a finished tune over the FULL E1 + E2 (never a --limit smoke run)."""
    if not isinstance(tuning, dict):
        return False
    n1, n2 = sizes or _full_selection_sizes()
    winner = tuning.get("winner") or {}
    seg = tuning.get("segmentation") or {}
    return bool(
        tuning.get("limit_e1") is None
        and tuning.get("limit_e2") is None
        and tuning.get("n_e1") == n1
        and tuning.get("n_e2") == n2
        and {"alpha", "beam", "segment_threshold"} <= set(winner)
        and seg.get("best") in (seg.get("scores") or {})
    )


def run_tune_step(model_dir: Path, out_path: Path, batch_size: int = EVAL_BATCH_SIZE) -> bool:
    """nmt.tune.run_tune over full E1 + E2 (PREREG §1); skipped when `out_path` is a valid full
    tune. Returns True when it ran."""
    if tuning_is_full(_read_json(out_path)):
        print(f"tune {Path(out_path).stem}: SKIP (valid full tuning at {out_path})")
        return False
    from nmt.tune import run_tune

    result = run_tune(Path(model_dir), Path(out_path), batch_size=batch_size)
    w = result["winner"]
    print(
        f"tune: winner alpha={w['alpha']} beam={w['beam']} "
        f"segment_threshold={w['segment_threshold']}"
    )
    return True


def candidate_objective(tuning: dict[str, Any]) -> dict[str, Any]:
    """The candidate's best selection objective: the segmentation stage's best-scoring entry
    (E1 fixed, T tuned on E2), scored by nmt.selection.selection_objective inside nmt.tune."""
    seg = tuning["segmentation"]
    scores = seg["scores"][seg["best"]]
    return {
        **{k: scores[k] for k in ("objective", "bleu_union", "chrf_union", "chrf_e1")},
        "config": dict(tuning["winner"]),
        "model_sha256": tuning.get("model_sha256"),
        "n_e1": tuning["n_e1"],
        "n_e2": tuning["n_e2"],
    }


def select_winner(tunings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick the (candidate, decode config) with the highest selection objective. `tunings` keeps
    the pre-registered candidate order; ties follow TIE_RULE. Refuses a non-full tuning."""
    if not tunings:
        raise EvalStepError("select: no candidates")
    sizes = _full_selection_sizes()
    for name, tuning in tunings.items():
        if not tuning_is_full(tuning, sizes):
            raise EvalStepError(f"select: tuning for candidate {name!r} is missing or not full")
    cands = {name: candidate_objective(t) for name, t in tunings.items()}
    best = max(c["objective"] for c in cands.values())
    tied = [n for n, c in cands.items() if best - c["objective"] <= TIE_EPSILON]
    winner = tied[0]
    return {
        "objective_formula": "0.4*BLEU(E1+E2) + 0.4*chrF(E1+E2) + 0.2*chrF(E1), official score.py",
        "selection_sets": "E1 + E2 only (nmt.selection)",
        "candidate_order": list(tunings),
        "candidates": cands,
        "tie_rule": TIE_RULE,
        "tied_candidates": tied,
        "winner": {
            "candidate": winner,
            "objective": cands[winner]["objective"],
            **cands[winner]["config"],
        },
    }


def run_select(tuning_dir: Path, candidates: Sequence[str], out_path: Path) -> dict[str, Any]:
    """Write selection.json from `<tuning_dir>/<candidate>.json` for every candidate."""
    tunings: dict[str, dict[str, Any]] = {}
    for name in candidates:
        data = _read_json(Path(tuning_dir) / f"{name}.json")
        if not isinstance(data, dict):
            raise EvalStepError(f"select: {tuning_dir}/{name}.json is missing or unreadable")
        tunings[name] = data
    result = select_winner(tunings)
    _write_json(out_path, result)
    w = result["winner"]
    print(
        f"select: winner {w['candidate']} objective={w['objective']:.4f} alpha={w['alpha']} "
        f"beam={w['beam']} segment_threshold={w['segment_threshold']}"
    )
    return result


def selection_is_valid(path: Path, candidates: Sequence[str]) -> bool:
    data = _read_json(path)
    return bool(
        isinstance(data, dict)
        and data.get("candidate_order") == list(candidates)
        and (data.get("winner") or {}).get("candidate") in candidates
    )


# ---------------------------------------------------------------------------------------------
# decode (official dev, E1, E2, E2-synth, E3; test for main)
# ---------------------------------------------------------------------------------------------


def prediction_file_valid(path: Path, expected_ids: Sequence[str]) -> bool:
    """A JSON object keyed by exactly `expected_ids` whose values are non-empty strings."""
    data = _read_json(path)
    return bool(
        isinstance(data, dict)
        and set(data) == set(expected_ids)
        and len(data) == len(expected_ids)
        and all(isinstance(v, str) and v.strip() for v in data.values())
    )


def _stats(translator: Any) -> tuple[int, int, int]:
    s = translator.stats
    return s.n_beam, s.n_greedy_fallback, s.n_copy_fallback


def _decode_to_file(
    translator: Any,
    rows: list[dict[str, Any]],
    path: Path,
    *,
    beam: int,
    alpha: float,
    segment_threshold: int | None,
    batch_size: int,
) -> dict[str, Any]:
    before, t0 = _stats(translator), time.monotonic()
    outputs = translator.translate(
        [r["source"] for r in rows],
        batch_size=batch_size,
        beam=beam,
        alpha=alpha,
        segment_threshold=segment_threshold,
    )
    after = _stats(translator)
    _write_json(path, dict(zip((r["id"] for r in rows), outputs, strict=True)))
    deltas = [a - b for a, b in zip(after, before, strict=True)]
    return {
        "n": len(rows),
        "seconds": round(time.monotonic() - t0, 3),
        "fallback_counts": dict(zip(("beam", "greedy", "copy"), deltas, strict=True)),
    }


def run_decode(
    translator: Any,
    selection: dict[str, Any],
    eval_dir: Path,
    *,
    include_test: bool,
    batch_size: int = EVAL_BATCH_SIZE,
    load_rows: Callable[[str], list[dict[str, Any]]] | None = None,
    test_inputs: Path | None = None,
) -> dict[str, Any]:
    """Decode every split twice with the selected alpha/beam: segmentation OFF, and at the tuned T
    (when T is "off" the second variant is a byte copy of the first, recorded as identical), plus
    the test set at the full tuned config for the main run. Each output file is skipped when it
    already validates, so a resumed session redoes only what is missing. Writes decode_summary.json.
    """
    from nmt.evaluate import load_split

    load = load_rows or (lambda split: load_split(split)[0])
    win = selection["winner"]
    beam, alpha, threshold = win["beam"], win["alpha"], win["segment_threshold"]
    eval_dir = Path(eval_dir)
    summary_path = eval_dir / "decode_summary.json"
    config = {"alpha": alpha, "beam": beam, "segment_threshold": threshold}
    prior = _read_json(summary_path)
    if isinstance(prior, dict) and (prior.get("candidate"), prior.get("config")) == (
        win["candidate"],
        config,
    ):
        meta: dict[str, Any] = prior.get("files", {})
    else:
        # Outputs of a different candidate/config (or of an unknown one) must never be reused.
        if (eval_dir / "predictions").exists() or (eval_dir / "test_predictions.json").exists():
            print("decode: predictions from another candidate/config found; discarding them")
        shutil.rmtree(eval_dir / "predictions", ignore_errors=True)
        (eval_dir / "test_predictions.json").unlink(missing_ok=True)
        meta = {}
    summary: dict[str, Any] = {
        "candidate": win["candidate"],
        "config": config,
        "tuned_threshold_is_off": threshold is None,
        "batch_size": batch_size,
        "variants_identical": threshold is None,
        "files": meta,
    }
    _write_json(summary_path, summary)  # config first: a crash can never orphan its predictions

    def one(rel: str, rows: list[dict[str, Any]], seg: int | None) -> None:
        path = eval_dir / rel
        if prediction_file_valid(path, [r["id"] for r in rows]):
            print(f"decode {rel}: SKIP (valid)")
            return
        meta[rel] = _decode_to_file(
            translator,
            rows,
            path,
            beam=beam,
            alpha=alpha,
            segment_threshold=seg,
            batch_size=batch_size,
        )
        _write_json(summary_path, summary)  # progress survives a disconnect
        print(f"decode {rel}: {meta[rel]['n']} sentences in {meta[rel]['seconds']}s")

    for split in DECODE_SPLITS:
        rows = load(split)
        off_rel = f"predictions/{VARIANT_SEG_OFF}/{split}_predictions.json"
        tuned_rel = f"predictions/{VARIANT_TUNED}/{split}_predictions.json"
        one(off_rel, rows, None)
        if threshold is None:
            ids = [r["id"] for r in rows]
            if not prediction_file_valid(eval_dir / tuned_rel, ids):
                (eval_dir / tuned_rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(eval_dir / off_rel, eval_dir / tuned_rel)
                meta[tuned_rel] = {"copy_of": off_rel, "n": len(rows)}
        else:
            one(tuned_rel, rows, threshold)
    if include_test:
        path = test_inputs or REPO_ROOT / "data" / "test" / "inputs.jsonl"
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln]
        one("test_predictions.json", rows, threshold)
    summary["splits"] = list(DECODE_SPLITS)
    summary["test_included"] = include_test
    _write_json(summary_path, summary)
    return summary


# ---------------------------------------------------------------------------------------------
# validate-test
# ---------------------------------------------------------------------------------------------


def validate_test_predictions(
    pred_path: Path, out_path: Path, sample_path: Path | None = None
) -> dict[str, Any]:
    """Exactly the 330 ids of data/test/sample_submission.json, 0 empty strings, a UTF-8 JSON
    object (nmt.submission.validate_submission). Writes validation.json and raises on any failure
    (after recording `valid: false` with the reason)."""
    from nmt import submission

    sample = Path(sample_path) if sample_path else submission.DEFAULT_SAMPLE_PATH
    record: dict[str, Any] = {
        "pred": str(pred_path),
        "sample": str(sample),
        "sample_sha256": _sha256_file(sample),
    }
    try:
        info = submission.validate_submission(pred_path, sample)
    except (ValueError, OSError) as exc:  # includes UnicodeDecodeError and JSONDecodeError
        record.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        _write_json(out_path, record)
        raise EvalStepError(f"test predictions INVALID: {exc}") from exc
    record.update(
        {
            "valid": True,
            "n_ids": info["n_ids"],
            "empty_strings": 0,
            "pred_sha256": _sha256_file(pred_path),
        }
    )
    _write_json(out_path, record)
    print(f"validate-test: OK ({info['n_ids']} ids, 0 empty strings)")
    return record


# ---------------------------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------------------------

_MODEL_FILES = ("model.safetensors", "config.json", "spm.model")


def expected_candidates(run: str) -> tuple[str, ...]:
    """The candidate names a finished `run` must have tuned (pre-registered)."""
    return MAIN_CANDIDATES if run == "main" else ABLATION_CANDIDATES


def required_rels(run: str, candidate_order: Sequence[str]) -> list[str]:
    """Paths under runs/<run>/ that a complete upload holds (manifest.json excluded)."""
    rels = ["bench.json", "selection.json", "decode_summary.json", "run_meta.json"]
    rels += [f"tuning/{c}.json" for c in candidate_order]
    rels += [f"predictions/{v}/{s}_predictions.json" for v in VARIANTS for s in DECODE_SPLITS]
    if run == "main":
        rels += ["test_predictions.json", "validation.json"]
    return rels + [f"model/{name}" for name in _MODEL_FILES]


def collect_upload_files(eval_dir: Path, run: str) -> list[tuple[Path, str]]:
    """(local path, path in repo) for everything the local pipeline needs, under runs/<run>/.
    Raises EvalStepError listing any required file that is missing."""
    eval_dir = Path(eval_dir)
    selection = _read_json(eval_dir / "selection.json")
    if not isinstance(selection, dict):
        raise EvalStepError(f"upload: {eval_dir}/selection.json missing or unreadable")
    winner = selection["winner"]["candidate"]
    files = []
    for rel in required_rels(run, selection["candidate_order"]):
        if rel.startswith("model/"):
            local = eval_dir / "candidates" / winner / rel.removeprefix("model/")
        else:
            local = eval_dir / rel
        files.append((local, f"runs/{run}/{rel}"))
    missing = [str(p) for p, _ in files if not p.is_file()]
    if missing:
        raise EvalStepError("upload: missing required file(s): " + ", ".join(missing))
    return files


def commit_message(run: str) -> str:
    """The single upload commit's message; hf-verify looks for exactly this title."""
    return f"eval_l4: {run} predictions, tuning, selection, selected model"


def build_manifest(run: str, files: Sequence[tuple[Path, str]], eval_dir: Path) -> dict[str, Any]:
    """sha256 + size of every uploaded file (keys relative to runs/<run>/), plus the selection's
    candidate order and winner. Uploaded as runs/<run>/manifest.json in the same commit."""
    selection = _read_json(Path(eval_dir) / "selection.json") or {}
    prefix = f"runs/{run}/"
    return {
        "schema": 1,
        "run": run,
        "candidate_order": list(selection.get("candidate_order", [])),
        "winner": (selection.get("winner") or {}).get("candidate"),
        "files": {
            rel.removeprefix(prefix): {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p, rel in files
        },
    }


def _lfs_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def verify_run_on_hf(api: Any, repo_id: str, run: str) -> dict[str, Any]:
    """Evidence from the private HF repo itself that `run` was uploaded completely.

    Complete means ALL of: the repo is private; a commit titled `commit_message(run)` exists; at
    the repo head runs/<run>/manifest.json exists, names this run, lists every required file
    (candidates fixed by PREREG); every listed file exists with the manifest's size and sha256
    (LFS files via the hub's own sha256, small files by downloading and hashing them); for main,
    validation.json says valid with 330 ids. Returns {"complete": bool, "reason", "revision"}
    where revision is the upload commit's sha. A 404 (no repo / no runs/<run>/) is "not
    complete"; any other failure to READ the evidence raises EvalStepError (fail closed: never a
    silent skip), and a non-private repo raises HFNotPrivateError.
    """

    def no(reason: str) -> dict[str, Any]:
        return {"complete": False, "reason": reason, "revision": None}

    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        if _status(exc) == 404:
            return no(f"repo {repo_id} does not exist")
        raise EvalStepError(
            f"hf-verify: cannot read {repo_id} ({type(exc).__name__}: {exc})"
        ) from exc
    if getattr(info, "private", None) is not True:
        raise HFNotPrivateError(
            f"hf-verify: HF repo {repo_id} is NOT private "
            f"(private={getattr(info, 'private', None)!r})"
        )
    head = getattr(info, "sha", None)
    if not head or not _REVISION_RE.match(str(head)):
        raise EvalStepError(f"hf-verify: {repo_id} reported no usable head revision ({head!r})")

    prefix = f"runs/{run}"
    try:
        entries = list(
            api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=prefix,
                recursive=True,
                repo_type="model",
                revision=head,
            )
        )
        commits = list(api.list_repo_commits(repo_id=repo_id, repo_type="model", revision=head))
    except Exception as exc:
        if _status(exc) == 404:
            return no(f"nothing under {prefix}/ in {repo_id}")
        raise EvalStepError(
            f"hf-verify: cannot list {repo_id} ({type(exc).__name__}: {exc})"
        ) from exc
    upload = next((c for c in commits if getattr(c, "title", None) == commit_message(run)), None)
    if upload is None:
        return no(f"no commit titled {commit_message(run)!r}")
    remote = {e.path: e for e in entries if getattr(e, "size", None) is not None}

    def fetch(rel: str) -> Path:
        try:
            return Path(
                api.hf_hub_download(
                    repo_id=repo_id, filename=f"{prefix}/{rel}", repo_type="model", revision=head
                )
            )
        except Exception as exc:
            raise EvalStepError(f"hf-verify: cannot download {prefix}/{rel} ({exc})") from exc

    if f"{prefix}/{MANIFEST_NAME}" not in remote:
        return no(f"{prefix}/{MANIFEST_NAME} is missing")
    manifest = _read_json(fetch(MANIFEST_NAME))
    if not isinstance(manifest, dict) or manifest.get("run") != run:
        return no("manifest.json is unreadable or names another run")
    listed = manifest.get("files")
    if manifest.get("candidate_order") != list(expected_candidates(run)) or not isinstance(
        listed, dict
    ):
        return no("manifest.json candidate order / file list is not the pre-registered one")
    absent = [r for r in required_rels(run, expected_candidates(run)) if r not in listed]
    if absent:
        return no(f"manifest.json lacks required file(s): {', '.join(absent)}")
    for rel, want in listed.items():
        entry = remote.get(f"{prefix}/{rel}")
        if entry is None:
            return no(f"{prefix}/{rel} is listed in the manifest but absent from the repo")
        if entry.size != want.get("bytes"):
            return no(f"{prefix}/{rel}: size {entry.size} != manifest {want.get('bytes')}")
        got = _lfs_sha256(entry) or _sha256_file(fetch(rel))
        if got != want.get("sha256"):
            return no(f"{prefix}/{rel}: sha256 differs from the manifest")
    if run == "main":
        validation = _read_json(fetch("validation.json"))
        if not (
            isinstance(validation, dict)
            and validation.get("valid") is True
            and validation.get("n_ids") == EXPECTED_TEST_IDS
        ):
            return no(f"validation.json does not report valid / {EXPECTED_TEST_IDS} ids")
    return {
        "complete": True,
        "reason": "manifest and sha256 verified",
        "revision": upload.commit_id,
    }


def upload_run(api: Any, repo_id: str, run: str, eval_dir: Path) -> dict[str, Any]:
    """Upload one run to the PRIVATE HF repo in a single commit. Order: gather files (refuse if
    any missing), check token scope, look for evidence on HF that the run is already complete
    (verify_run_on_hf; skip if so), create_repo(private=True, exist_ok=True) and read back
    (refuse unless private), commit (files + manifest.json), read back privacy again. Prints
    `HF_EVAL_REVISION=<sha>` and records it in hf_upload.json."""
    from huggingface_hub import CommitOperationAdd

    eval_dir = Path(eval_dir)
    if run not in RUNS:
        raise EvalStepError(f"upload: RUN {run!r} not in {RUNS}")
    if not _REPO_ID_RE.match(repo_id):
        raise EvalStepError(f"HF_EVAL_REPO={repo_id!r} is not '<owner>/<name>'")
    files = collect_upload_files(eval_dir, run)
    check_write_token(api, repo_id)
    state = verify_run_on_hf(api, repo_id, run)  # also refuses a non-private repo
    if state["complete"]:
        assert_repo_private(api, repo_id, "upload already done")
        print(f"upload: SKIP (verified complete on HF at revision {state['revision']})")
        print(f"HF_EVAL_REVISION={state['revision']}")
        record = {
            "repo": repo_id,
            "run": run,
            "revision": state["revision"],
            "private": True,
            "files": [r for _, r in files],
            "verified_existing": True,
        }
        _write_json(eval_dir / "hf_upload.json", record)
        return record
    ensure_private_repo(api, repo_id)
    manifest = build_manifest(run, files, eval_dir)
    _write_json(eval_dir / MANIFEST_NAME, manifest)
    operations = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in files]
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"runs/{run}/{MANIFEST_NAME}",
            path_or_fileobj=str(eval_dir / MANIFEST_NAME),
        )
    )
    try:
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=commit_message(run),
        )
    except Exception as exc:
        if _is_auth_error(exc):
            raise HFWriteTokenError(write_token_message(repo_id, f"upload refused: {exc}")) from exc
        raise
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_oid", None)
    assert_repo_private(api, repo_id, "after upload")
    if not revision or not _REVISION_RE.match(str(revision)):
        raise EvalStepError(f"upload: commit returned no usable revision sha ({revision!r})")
    record = {
        "repo": repo_id,
        "run": run,
        "revision": revision,
        "private": True,
        "files": [r for _, r in files] + [f"runs/{run}/{MANIFEST_NAME}"],
        "manifest_sha256": _sha256_file(eval_dir / MANIFEST_NAME),
    }
    _write_json(eval_dir / "hf_upload.json", record)
    print(f"HF_EVAL_REVISION={revision}")
    return record


# ---------------------------------------------------------------------------------------------
# workload (inputs of the notebook's runtime estimate)
# ---------------------------------------------------------------------------------------------


def measure_workload() -> dict[str, Any]:
    """Token counts the notebook's estimate multiplies by a decode rate. Output tokens are proxied
    by the reference's SentencePiece token count (the test set has no references: its source
    tokens times the ref/src ratio measured on the other splits)."""
    import sentencepiece as spm

    from nmt.evaluate import load_split

    tok = REPO_ROOT / "tokenizer" / "spm.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(tok))
    count = lambda texts: sum(len(sp.encode(t, out_type=int)) for t in texts)  # noqa: E731
    splits: dict[str, dict[str, Any]] = {}
    for name in ("e1", "e2", "e3", "e2synth", "dev"):
        inputs, labels = load_split(name)
        splits[name] = {
            "n": len(inputs),
            "src_tokens": count(r["source"] for r in inputs),
            "out_tokens": count(r["reference"] for r in labels),
        }
    ratio = sum(s["out_tokens"] for s in splits.values()) / sum(
        s["src_tokens"] for s in splits.values()
    )
    test_rows = [
        json.loads(ln)
        for ln in (REPO_ROOT / "data" / "test" / "inputs.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    test_src = count(r["source"] for r in test_rows)
    splits["test"] = {
        "n": len(test_rows),
        "src_tokens": test_src,
        "out_tokens": round(test_src * ratio),
    }
    e2_rows = load_split("e2")[1][:BENCH_N_SENTENCES]
    return {
        "tokenizer_sha256": _sha256_file(tok),
        "out_over_src_token_ratio": round(ratio, 4),
        "bench_e2_first200_out_tokens": count(r["reference"] for r in e2_rows),
        "splits": splits,
    }


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _hf_api() -> Any:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")  # the name huggingface_hub reads
    if not token:
        raise EvalStepError(
            "HF_TOKEN is not set in the environment. In Colab the notebook's secrets cell exports "
            "your HF_TOKEN_WRITE secret under this name (Notebook access on); re-run that cell. "
            "Outside Colab, export a write-scoped token as HF_TOKEN. The value is never printed."
        )
    return HfApi(token=token)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "eval_l4 steps")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hf-check", help="fail fast: token can write, repo (if any) is private")
    s.add_argument("--repo", required=True)

    s = sub.add_parser(
        "hf-verify", help="is runs/<run>/ complete on the private HF repo (manifest + sha256)?"
    )
    s.add_argument("--repo", required=True)
    s.add_argument("--run", required=True, choices=RUNS)

    s = sub.add_parser("candidates", help="average checkpoints and export candidate model dirs")
    s.add_argument("--ckpt-dir", required=True, type=Path)
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--out-dir", required=True, type=Path, help="parent of <name>/ export dirs")
    s.add_argument("--candidate", action="append", required=True, help="name=step,step,...")
    s.add_argument("--only", default=None, help="build just this candidate (all are checked)")

    s = sub.add_parser("bench", help="throughput benchmark (first 200 E2 sentences)")
    s.add_argument("--model", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--run", required=True, choices=RUNS)
    s.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)

    s = sub.add_parser("tune", help="PREREG §1 decoding tuning on full E1+E2 for one candidate")
    s.add_argument("--model", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)

    s = sub.add_parser("select", help="pick the (candidate, decode config) by the §2 objective")
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--candidates", nargs="+", required=True, help="in pre-registered order")
    s.add_argument("--out", required=True, type=Path)

    s = sub.add_parser("decode", help="decode dev/E1/E2/E2-synth/E3 (+test) for the winner")
    s.add_argument("--eval-dir", required=True, type=Path)
    s.add_argument("--test", action="store_true", help="also decode the test set (main only)")
    s.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)

    s = sub.add_parser("validate-test", help="validate test_predictions.json")
    s.add_argument("--pred", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)

    s = sub.add_parser("upload", help="upload one run to the PRIVATE HF repo")
    s.add_argument("--repo", required=True)
    s.add_argument("--run", required=True, choices=RUNS)
    s.add_argument("--eval-dir", required=True, type=Path)

    s = sub.add_parser("workload", help="write colab/eval_workload.json")
    s.add_argument("--out", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except EvalStepError as exc:
        print(f"eval_l4 {args.cmd}: FAILED: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "hf-check":
        info = hf_check(_hf_api(), args.repo)
        print(
            f"hf-check: OK account={info['account']} repo={args.repo} exists={info['exists']} "
            f"private=True probe: {info['probe']} ({PROBE_PATH})"
        )
    elif args.cmd == "hf-verify":
        state = verify_run_on_hf(_hf_api(), args.repo, args.run)
        # One machine-readable line (the notebook parses it) then the human reason.
        print(f"HF_RUN_COMPLETE={'true' if state['complete'] else 'false'}")
        if state["revision"]:
            print(f"HF_EVAL_REVISION={state['revision']}")
        print(f"hf-verify {args.run}: {state['reason']}")
    elif args.cmd == "candidates":
        specs = parse_candidate_specs(args.candidate)
        check_candidate_files(args.ckpt_dir, specs)
        if args.only is not None and args.only not in specs:
            raise EvalStepError(f"--only {args.only!r} is not one of {list(specs)}")
        for name, steps in specs.items():
            if args.only in (None, name):
                build_candidate(name, steps, args.ckpt_dir, args.config, args.out_dir / name)
    elif args.cmd == "bench":
        if bench_is_valid(args.out):
            print(f"bench: SKIP (valid {args.out})")
        else:
            r = run_bench(args.model, args.out, args.run, args.batch_size)
            for label, m in r["modes"].items():
                print(
                    f"bench {label}: {m['sentences_per_second']} sent/s "
                    f"{m['output_tokens_per_second']} out-tok/s on {r['gpu']}"
                )
    elif args.cmd == "tune":
        run_tune_step(args.model, args.out, args.batch_size)
    elif args.cmd == "select":
        if selection_is_valid(args.out, args.candidates) and _selection_current(args):
            print(f"select: SKIP (valid {args.out})")
        else:
            run_select(args.tuning_dir, args.candidates, args.out)
    elif args.cmd == "decode":
        _cmd_decode(args)
    elif args.cmd == "validate-test":
        validate_test_predictions(args.pred, args.out)
    elif args.cmd == "upload":
        upload_run(_hf_api(), args.repo, args.run, args.eval_dir)
    elif args.cmd == "workload":
        _write_json(args.out, measure_workload())
    return 0


def _selection_current(args: argparse.Namespace) -> bool:
    """selection.json is only reused if no tuning file is newer than it."""
    out = Path(args.out)
    return all(
        (Path(args.tuning_dir) / f"{c}.json").stat().st_mtime <= out.stat().st_mtime
        for c in args.candidates
    )


def _cmd_decode(args: argparse.Namespace) -> None:
    from nmt.translate import Translator

    selection = _read_json(args.eval_dir / "selection.json")
    if not isinstance(selection, dict):
        raise EvalStepError(f"decode: {args.eval_dir}/selection.json missing or unreadable")
    model_dir = args.eval_dir / "candidates" / selection["winner"]["candidate"]
    meta = _read_json(model_dir / "candidate_meta.json")
    if not isinstance(meta, dict) or not candidate_is_valid(model_dir, meta.get("steps", [])):
        raise EvalStepError(f"decode: selected candidate export {model_dir} is not valid")
    translator = Translator.from_pretrained(str(model_dir))
    run_decode(
        translator, selection, args.eval_dir, include_test=args.test, batch_size=args.batch_size
    )


if __name__ == "__main__":
    raise SystemExit(main())
