from __future__ import annotations

# The `final_all` selection run (PREREG 2026-10-02 "post-selection amendment", rules 1-3, 5, 6):
# the exhaustive candidate set
#
#   {main final, branch A, branch B, {main,A}, {main,B}, {A,B}, {main,A,B}}
#     x {beam (rule-1 grid), MBR beam n-best N=8, N=16, MBR epsilon sampling (0.02) N=8, N=16}
#
# = 7 x 5 = 35 candidates, each tuned on the FULL E1 + E2 with the section 2 objective
# (nmt.selection via nmt.tune), then one winner (ties to the earlier-listed candidate), decoded on
# dev / E1 / E2 / E2-synth / E3 (+ the 330-id test set) and uploaded with a manifest under the run
# name `final_all`. PREREG fixes the set: nothing here prunes it. `estimate_final_all` prices it.
#
# Order of the candidate list (the tie-break order): model-set major in the order above, decode
# kind minor (beam, mbr_beam8, mbr_beam16, mbr_eps0.02_n8, mbr_eps0.02_n16).
#
# Interpretations of the rule text (also in the PR description, none changes the candidate set):
#   * "Beam settings inside a pool use the winner of rule 1 for that model": an MBR candidate takes
#     the GNMT alpha of the winning (alpha, beam) of the SAME model-set's `beam` candidate. The
#     pool size N, not that beam width, sets the beam width of the n-best pool (an N-best list
#     needs beam >= N). Sampling pools do not use alpha. The segmentation threshold T is then tuned
#     for the MBR candidate exactly like rule 1's T step (E1 undecoded-segmented, E2 per T).
#   * Every beam candidate runs the unchanged nmt.tune procedure on the rule-1 grid
#     alpha {1.2,1.4,1.6,1.8,2.0} x beam {1,4,5}, then the T step.
#
# Step CLI (each step is idempotent: output written last, skipped when it validates):
#   python -m nmt.final_all tune|select|decode|plan|estimate ...
# The planning function `plan_final_all` returns the ordered (step name, argv) list; this module's
# top level imports only the stdlib (the Colab kernel rule): torch and the model code are imported
# inside the step functions, which run in their own subprocesses.
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmt import eval_l4 as ev

FINAL_ALL_ALPHAS: tuple[float, ...] = (1.2, 1.4, 1.6, 1.8, 2.0)  # PREREG rule 1
FINAL_ALL_BEAMS: tuple[int, ...] = (1, 4, 5)  # the unchanged section 1 beam grid
MODEL_NAMES: tuple[str, ...] = ("main", "A", "B")
# PREREG rules 3 and 5: the single models, then exactly the four ensembles, in this order.
MODEL_SETS: tuple[tuple[str, ...], ...] = (
    ("main",),
    ("A",),
    ("B",),
    ("main", "A"),
    ("main", "B"),
    ("A", "B"),
    ("main", "A", "B"),
)
# (kind, N); kind "beam" = n-best from a beam of width N, "sample" = N epsilon samples.
POOL_SPECS: tuple[tuple[str, int], ...] = (("beam", 8), ("beam", 16), ("sample", 8), ("sample", 16))
SAMPLING_EPSILON = 0.02  # PREREG rule 2 (== nmt.mbr.DEFAULT_EPSILON, checked by a test)
SAMPLING_SEED = 1234  # PREREG rule 2 (== nmt.mbr.DEFAULT_SAMPLING_SEED)
BEAM_KIND = "beam"
FINAL_ALL_TIE_RULE = (
    f"candidates within {ev.TIE_EPSILON:g} of the best objective are tied; the tie goes to the "
    "earliest candidate in the pre-registered order (model-set major: main, A, B, main+A, main+B, "
    "A+B, main+A+B; decode kind minor: beam, mbr_beam8, mbr_beam16, mbr_eps0.02_n8, "
    "mbr_eps0.02_n16)"
)


def pool_label(kind: str, n: int) -> str:
    """The decode-kind part of a candidate name for the MBR pool (kind, n)."""
    return f"mbr_beam{n}" if kind == "beam" else f"mbr_eps{SAMPLING_EPSILON:g}_n{n}"


@dataclass(frozen=True)
class FinalAllCandidate:
    """One of the 35 candidates: which models, and beam search (`pool is None`) or an MBR pool."""

    name: str
    members: tuple[str, ...]
    pool: tuple[str, int] | None

    @property
    def beam_name(self) -> str:
        """The name of this model-set's plain-beam candidate (an MBR candidate's alpha source)."""
        return candidate_name(self.members, None)

    @property
    def mbr(self) -> dict[str, Any] | None:
        if self.pool is None:
            return None
        kind, n = self.pool
        return {"kind": kind, "n": n, "epsilon": SAMPLING_EPSILON, "seed": SAMPLING_SEED}


def candidate_name(members: Sequence[str], pool: tuple[str, int] | None) -> str:
    kind = BEAM_KIND if pool is None else pool_label(*pool)
    return "+".join(members) + "__" + kind


def final_all_candidates() -> list[FinalAllCandidate]:
    """All 35 candidates in the pre-registered (tie-break) order."""
    pools: list[tuple[str, int] | None] = [None, *POOL_SPECS]
    return [FinalAllCandidate(candidate_name(m, p), m, p) for m in MODEL_SETS for p in pools]


FINAL_ALL_CANDIDATES: tuple[str, ...] = tuple(c.name for c in final_all_candidates())


def candidate_by_name(name: str) -> FinalAllCandidate:
    for c in final_all_candidates():
        if c.name == name:
            return c
    raise ev.EvalStepError(f"unknown final_all candidate {name!r}")


def parse_model_args(specs: Sequence[str]) -> dict[str, Path]:
    """`["main=DIR", "A=DIR", "B=DIR"]` -> {"main": Path, ...}; refuses unknown/duplicate names."""
    out: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ev.EvalStepError(f"bad --model {spec!r}; expected name=DIR")
        if name not in MODEL_NAMES:
            raise ev.EvalStepError(f"--model name {name!r} is not one of {MODEL_NAMES}")
        if name in out:
            raise ev.EvalStepError(f"duplicate --model {name!r}")
        out[name] = Path(path)
    return out


def check_model_dirs(models: Mapping[str, Path], members: Sequence[str]) -> None:
    """Every member must have a finished HF-style export (weights, config, tokenizer)."""
    for m in members:
        if m not in models:
            raise ev.EvalStepError(f"no --model {m}=DIR given")
        missing = [f for f in ev._MODEL_FILES if not (Path(models[m]) / f).is_file()]
        if missing:
            raise ev.EvalStepError(f"model {m} ({models[m]}) lacks {', '.join(missing)}")


# ---------------------------------------------------------------------------------------------
# translators
# ---------------------------------------------------------------------------------------------


class PoolTranslator:
    """A Translator whose `translate` always decodes with one MBR pool, so nmt.tune's helpers and
    `eval_l4.run_decode` (which call `translate(beam=, alpha=, segment_threshold=)`) can drive MBR
    unchanged. `beam` is ignored; everything else is delegated to the wrapped translator."""

    def __init__(self, translator: Any, mbr: Any) -> None:
        self._inner = translator
        self._mbr = mbr

    def translate(self, texts: list[str], **kwargs: Any) -> list[str]:
        kwargs.pop("beam", None)
        return self._inner.translate(texts, mbr=self._mbr, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_translator(
    models: Mapping[str, Path],
    members: Sequence[str],
    mbr: Mapping[str, Any] | None = None,
    device: str | None = None,
) -> Any:
    """Translator for one model (plain) or several (nmt.ensemble), wrapped in a PoolTranslator when
    `mbr` (a dict of MBRConfig fields) is given."""
    check_model_dirs(models, members)
    dirs = [models[m] for m in members]
    if len(dirs) == 1:
        from nmt.translate import Translator

        translator = Translator.from_pretrained(str(dirs[0]), device=device)
    else:
        from nmt.ensemble import load_ensemble_translator

        translator = load_ensemble_translator(dirs, device=device)
    if mbr is None:
        return translator
    from nmt.mbr import MBRConfig

    return PoolTranslator(translator, MBRConfig(**mbr))


# ---------------------------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------------------------


def grid_matches(tuning: Any) -> bool:
    """True iff a beam candidate's tuning file used exactly the rule-1 grid (alpha x beam)."""
    if not isinstance(tuning, dict):
        return False
    grid = (tuning.get("alpha_beam") or {}).get("grid") or []
    want = {(a, b) for a in FINAL_ALL_ALPHAS for b in FINAL_ALL_BEAMS}
    return {(g.get("alpha"), g.get("beam")) for g in grid} == want


def _tuning_ok(
    tuning: Any,
    cand: FinalAllCandidate,
    member_sha256: Mapping[str, str] | None = None,
    base_alpha: float | None = None,
) -> bool:
    """A finished full tuning of `cand` on the right grid. When `member_sha256` (the CURRENT
    weights' hashes) is given, the file must have been made by exactly those weights, and when
    `base_alpha` (the CURRENT beam winner's alpha) is given, an MBR file must have used that alpha;
    otherwise it is stale (a model or an upstream tuning changed) and is not accepted. A file that
    records no hash / alpha cannot be shown current, so it fails too."""
    if not ev.tuning_is_full(tuning):
        return False
    if member_sha256 is not None and tuning.get("member_sha256") != dict(member_sha256):
        return False
    if cand.pool is not None:
        if base_alpha is not None and (tuning.get("alpha_beam") or {}).get("alpha") != base_alpha:
            return False
        return tuning.get("candidate") == cand.name
    return grid_matches(tuning)


def _member_sha256(models: Mapping[str, Path], members: Sequence[str]) -> dict[str, str]:
    return {m: ev._sha256_file(Path(models[m]) / "model.safetensors") for m in members}


def run_pool_tune(
    translator: Any,
    cand: FinalAllCandidate,
    base_tuning: dict[str, Any],
    out_path: Path,
    batch_size: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Tune one MBR candidate on full E1 + E2: alpha fixed to `base_tuning`'s rule-1 winner, E1
    decoded once without segmentation, then the T step (nmt.tune._segmentation_tune) on E2. Writes
    a tuning file with the same schema as nmt.tune.run_tune (so selection reads it unchanged)."""
    from nmt.selection import load_selection_set
    from nmt.tune import (
        DEFAULT_SEG_THRESHOLDS,
        _decode_ids,
        _git_sha,
        _segmentation_tune,
    )

    e1, e2 = load_selection_set("e1"), load_selection_set("e2")
    alpha = base_tuning["winner"]["alpha"]
    assert cand.pool is not None
    n = cand.pool[1]
    e1_preds, e1_seconds = _decode_ids(
        translator, e1.ids, e1.sources, n, alpha, batch_size, segment_threshold=None
    )
    seg = _segmentation_tune(
        translator, e1, e2, e1_preds, alpha, n, DEFAULT_SEG_THRESHOLDS, batch_size
    )
    result: dict[str, Any] = {
        "model_dir": None,
        "model_sha256": None,
        "git_sha": _git_sha(),
        "limit_e1": None,
        "limit_e2": None,
        "n_e1": len(e1),
        "n_e2": len(e2),
        "batch_size": batch_size,
        "alpha_beam": {
            "note": "MBR candidate: alpha is the rule-1 winner of the beam candidate "
            f"{cand.beam_name}; no alpha x beam grid is searched",
            "alpha_source": cand.beam_name,
            "alpha": alpha,
            "e1_decode_seconds": round(e1_seconds, 3),
        },
        "segmentation": seg,
        "winner": {
            "alpha": alpha,
            # beam: the n-best pool's width for beam pools; sampling has no beam (None)
            "beam": n if cand.pool[0] == "beam" else None,
            "segment_threshold": seg["best_segment_threshold"],
            "mbr": cand.mbr,
        },
        "fallback_counts": {
            "beam": translator.stats.n_beam,
            "greedy": translator.stats.n_greedy_fallback,
            "copy": translator.stats.n_copy_fallback,
        },
        **extra,
    }
    ev._write_json(out_path, result)
    return result


def tune_candidate(
    name: str,
    models: Mapping[str, Path],
    tuning_dir: Path,
    batch_size: int = ev.EVAL_BATCH_SIZE,
    device: str | None = None,
) -> bool:
    """Tune one candidate into `<tuning_dir>/<name>.json`; skipped when a valid full tuning (on the
    right grid / for this candidate) exists. Returns True when it ran. An MBR candidate needs its
    model-set's beam tuning first (its alpha)."""
    cand = candidate_by_name(name)
    out = Path(tuning_dir) / f"{name}.json"
    check_model_dirs(models, cand.members)
    hashes = _member_sha256(models, cand.members)
    base = None
    if cand.pool is not None:
        base = ev._read_json(Path(tuning_dir) / f"{cand.beam_name}.json")
        if not _tuning_ok(base, candidate_by_name(cand.beam_name), hashes):
            raise ev.EvalStepError(
                f"tune {name}: needs a valid full rule-1 tuning of {cand.beam_name} made by the "
                "current model weights first"
            )
    base_alpha = base["winner"]["alpha"] if base else None
    existing = ev._read_json(out)
    if _tuning_ok(existing, cand, hashes, base_alpha):
        print(f"tune {name}: SKIP (valid full tuning at {out})")
        return False
    if existing is not None:
        print(f"tune {name}: existing {out} is stale or invalid for the current models; redoing")
    extra = {
        "candidate": name,
        "members": list(cand.members),
        "member_sha256": hashes,
        "mbr": cand.mbr,
    }
    translator = build_translator(models, cand.members, mbr=cand.mbr, device=device)
    if cand.pool is None:
        from nmt.tune import run_tune

        result = run_tune(
            Path(models[cand.members[0]]),
            out,
            alphas=FINAL_ALL_ALPHAS,
            beams=FINAL_ALL_BEAMS,
            batch_size=batch_size,
            translator=translator,
            extra=extra,
        )
    else:
        assert base is not None
        result = run_pool_tune(translator, cand, base, out, batch_size, extra)
    w = result["winner"]
    print(
        f"tune {name}: winner alpha={w['alpha']} beam={w['beam']} "
        f"segment_threshold={w['segment_threshold']}"
    )
    return True


# ---------------------------------------------------------------------------------------------
# select + decode
# ---------------------------------------------------------------------------------------------


def _check_consistent(tunings: Mapping[str, dict[str, Any]]) -> None:
    """Refuse a mix of stale and fresh tuning files (no model dirs are needed here): every
    candidate must record the same hash for the same member, and an MBR candidate's alpha must be
    its beam candidate's current winner alpha."""
    seen: dict[str, str] = {}
    for name, t in tunings.items():
        for member, sha in (t.get("member_sha256") or {}).items():
            if seen.setdefault(member, sha) != sha:
                raise ev.EvalStepError(
                    f"select: {name} was tuned with different weights for member {member!r} than "
                    "another candidate (a model changed between tuning steps); redo the tunes"
                )
    for c in final_all_candidates():
        if c.pool is not None:
            want = tunings[c.beam_name]["winner"]["alpha"]
            if (tunings[c.name].get("alpha_beam") or {}).get("alpha") != want:
                raise ev.EvalStepError(
                    f"select: {c.name} used a different alpha than the current winner of "
                    f"{c.beam_name} ({want}); redo its tune"
                )


def run_final_select(tuning_dir: Path, out_path: Path) -> dict[str, Any]:
    """selection.json over the 35 candidates (nmt.selection objective on E1 + E2): every
    candidate's objective and config, the winner (with its member list and MBR pool), the tie
    rule. Refuses a missing / non-full / wrong-grid tuning."""
    tunings: dict[str, dict[str, Any]] = {}
    for cand in final_all_candidates():
        data = ev._read_json(Path(tuning_dir) / f"{cand.name}.json")
        if not _tuning_ok(data, cand):
            raise ev.EvalStepError(
                f"select: {tuning_dir}/{cand.name}.json is missing, not a full E1+E2 tuning, or "
                "not on the pre-registered grid"
            )
        tunings[cand.name] = data
    _check_consistent(tunings)
    result = ev.select_winner(tunings)
    result["tie_rule"] = FINAL_ALL_TIE_RULE
    result["objective_formula"] += "; final_all: PREREG post-selection amendment rule 5"
    for name, entry in result["candidates"].items():
        entry["members"] = list(candidate_by_name(name).members)
    win = result["winner"]
    win["members"] = list(candidate_by_name(win["candidate"]).members)
    ev._write_json(out_path, result)
    print(
        f"select: winner {win['candidate']} objective={win['objective']:.4f} alpha={win['alpha']} "
        f"beam={win['beam']} segment_threshold={win['segment_threshold']} mbr={win.get('mbr')}"
    )
    return result


def final_selection_is_valid(path: Path, tuning_dir: Path) -> bool:
    """selection.json exists for the 35 candidates and no tuning file is newer than it."""
    p = Path(path)
    data = ev._read_json(p)
    if not (
        isinstance(data, dict)
        and data.get("candidate_order") == list(FINAL_ALL_CANDIDATES)
        and (data.get("winner") or {}).get("candidate") in FINAL_ALL_CANDIDATES
        and (data.get("winner") or {}).get("members")
    ):
        return False
    return all(
        (Path(tuning_dir) / f"{c}.json").stat().st_mtime <= p.stat().st_mtime
        for c in FINAL_ALL_CANDIDATES
    )


def decode_winner(
    eval_dir: Path,
    models: Mapping[str, Path],
    batch_size: int = ev.EVAL_BATCH_SIZE,
    include_test: bool = True,
    device: str | None = None,
    **run_decode_kwargs: Any,
) -> dict[str, Any]:
    """Decode dev / E1 / E2 / E2-synth / E3 (+ test) for the selected candidate at its tuned
    config, through eval_l4.run_decode (per-file resumable)."""
    selection = ev._read_json(Path(eval_dir) / "selection.json")
    if not isinstance(selection, dict) or not (selection.get("winner") or {}).get("members"):
        raise ev.EvalStepError(f"decode: {eval_dir}/selection.json missing or has no winner")
    win = selection["winner"]
    translator = build_translator(models, win["members"], mbr=win.get("mbr"), device=device)
    return ev.run_decode(
        translator,
        selection,
        Path(eval_dir),
        include_test=include_test,
        batch_size=batch_size,
        **run_decode_kwargs,
    )


# ---------------------------------------------------------------------------------------------
# plan (what the notebook's eval_plan does for the other runs)
# ---------------------------------------------------------------------------------------------


def _model_args(models: Mapping[str, Path]) -> list[str]:
    out: list[str] = []
    for name in MODEL_NAMES:
        out += ["--model", f"{name}={models[name]}"]
    return out


def plan_final_all(
    *,
    eval_root: Path,
    hf_repo: str,
    models: Mapping[str, Path],
    batch_size: int = ev.EVAL_BATCH_SIZE,
    python: str | None = None,
) -> list[tuple[str, list[str]]]:
    """The ordered (step name, argv) list of the `final_all` session, each argv a fresh
    interpreter. Same shape and order as the notebook's eval_plan: hf-verify (the notebook's skip
    check: it parses HF_RUN_COMPLETE and skips the rest when true), hf-check (token + private repo
    + write probe, BEFORE any GPU work), bench, the 35 tunings (an MBR candidate after its
    model-set's beam candidate, which the pre-registered order already guarantees), select,
    decode (+test), validate-test, and the PRIVATE upload last."""
    py = python or sys.executable
    root = Path(eval_root)
    base = [py, "-m", "nmt.eval_l4"]
    fa = [py, "-m", "nmt.final_all"]
    mdl = _model_args(models)
    batch = ["--batch-size", str(batch_size)]
    plan: list[tuple[str, list[str]]] = [
        ("hf-verify", [*base, "hf-verify", "--repo", hf_repo, "--run", ev.FINAL_ALL_RUN]),
        ("hf-check", [*base, "hf-check", "--repo", hf_repo]),
        (
            "bench",  # the main model, beam 5: the measured rates feed estimate_final_all
            [
                *base,
                "bench",
                "--model",
                str(models["main"]),
                "--out",
                str(root / "bench.json"),
                "--run",
                ev.FINAL_ALL_RUN,
                *batch,
            ],
        ),
    ]
    for name in FINAL_ALL_CANDIDATES:
        plan.append(
            (
                f"tune:{name}",
                [*fa, "tune", "--name", name, *mdl, "--tuning-dir", str(root / "tuning"), *batch],
            )
        )
    plan += [
        (
            "select",
            [
                *fa,
                "select",
                "--tuning-dir",
                str(root / "tuning"),
                "--out",
                str(root / "selection.json"),
            ],
        ),
        ("decode", [*fa, "decode", "--eval-dir", str(root), *mdl, *batch, "--test"]),
        (
            "validate-test",
            [
                *base,
                "validate-test",
                "--pred",
                str(root / "test_predictions.json"),
                "--out",
                str(root / "validation.json"),
            ],
        ),
        (
            "upload",
            [
                *base,
                "upload",
                "--repo",
                hf_repo,
                "--run",
                ev.FINAL_ALL_RUN,
                "--eval-dir",
                str(root),
                *mdl,
            ],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------------------------
# cost ESTIMATE
# ---------------------------------------------------------------------------------------------

# ASSUMED output-token decode rates of ONE model on the L4, used when no bench.json exists (the
# same assumption as the notebook's EVAL_ASSUMED_RATES; no GPU decode rate has been measured).
ASSUMED_RATES = {"greedy": 2500.0, "beam": 1000.0}
# MEASURED on the dev laptop CPU (not Colab, which may be slower): mean seconds of one mbr_select
# over a pool of N distinct strings of ~160 characters (the mean E1/E2 reference length), 300 pools
# each; the command and output are in the PR description.
MEASURED_MBR_CPU_SECONDS_PER_POOL = {8: 0.0120, 16: 0.0354}
ASSUMED_FIXED_SECONDS = 900.0  # same assumption as the notebook (clone, install, loads, upload)
CU_PER_HOUR = 1.54  # reported by GG for the L4 pilot


def estimate_final_all(
    workload: Mapping[str, Any],
    rates: Mapping[str, float] | None = None,
    basis: str = "ASSUMED rates (no bench yet)",
    mbr_cpu_seconds: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """ESTIMATE (not a measurement) of the wall time of the exhaustive 35-candidate session.

    Token counts come from colab/eval_workload.json (reference SentencePiece tokens as the proxy of
    output length). `rates` are single-model output tokens/s for greedy and beam 5 (bench.json, or
    ASSUMED). Cost model, each line an assumption:
      * an M-member ensemble costs M x a single model per decoded token (every step runs all
        members; the encoder cost is ignored);
      * time per output token is linear in beam width / sample count: beam k costs k/5 x the beam-5
        rate (the beam loop has per-row Python work, so this is plausibly close; an upper-bound
        flavour for GPU-parallel width), N samples cost like a beam of width N, beams 4 and 5 are
        both costed at the beam-5 rate (as the notebook's estimate does);
      * beam candidate: greedy once on E1+E2, 5 alphas x {4, 5} beams on E1+E2, T step = 5 E2
        decodes (rule 1, same procedure as the notebook estimate);
      * MBR candidate: E1+E2 once + 5 E2 decodes at the pool's cost, plus the chrF selection on CPU
        for each of those sentences (`mbr_cpu_seconds`, MEASURED once on a laptop CPU);
      * the winner is decoded on every split twice plus test; costed as the dearest candidate
        (3-member ensemble, N=16 sampling), an upper bound for this part.
    """
    rates = dict(rates or ASSUMED_RATES)
    cpu = dict(mbr_cpu_seconds or MEASURED_MBR_CPU_SECONDS_PER_POOL)
    sp = workload["splits"]
    e12 = sp["e1"]["out_tokens"] + sp["e2"]["out_tokens"]
    e2 = sp["e2"]["out_tokens"]
    n_sent_tuned = sp["e1"]["n"] + sp["e2"]["n"] + 5 * sp["e2"]["n"]
    beam_rate = rates["beam"]
    per_set: dict[str, dict[str, float]] = {}
    total_gpu = 0.0
    total_cpu = 0.0
    for members in MODEL_SETS:
        m = len(members)
        key = "+".join(members)
        beam_s = m * (
            e12 / rates["greedy"] + len(FINAL_ALL_ALPHAS) * 2 * e12 / beam_rate + 5 * e2 / beam_rate
        )
        parts = {"beam": beam_s}
        gpu = beam_s
        cpu_s = 0.0
        for kind, n in POOL_SPECS:
            pool_s = m * (e12 + 5 * e2) / (beam_rate * 5.0 / n)
            parts[pool_label(kind, n)] = pool_s
            gpu += pool_s
            pool_cpu = n_sent_tuned * cpu[n]
            parts[pool_label(kind, n) + "_chrf_cpu"] = pool_cpu
            cpu_s += pool_cpu
        per_set[key] = parts
        total_gpu += gpu
        total_cpu += cpu_s
    split_tokens = sum(sp[s]["out_tokens"] for s in ev.DECODE_SPLITS)
    final_tokens = 2 * split_tokens + sp["test"]["out_tokens"]  # two variants per split, test once
    worst = len(MODEL_NAMES) * final_tokens / (beam_rate * 5.0 / 16)
    final_sentences = 2 * sum(sp[s]["n"] for s in ev.DECODE_SPLITS) + sp["test"]["n"]
    worst_cpu = final_sentences * cpu[16]
    bench = workload["bench_e2_first200_out_tokens"] * (1 / rates["greedy"] + 1 / beam_rate)
    seconds = total_gpu + total_cpu + worst + worst_cpu + bench + ASSUMED_FIXED_SECONDS
    return {
        "basis": basis,
        "rates": rates,
        "n_candidates": len(FINAL_ALL_CANDIDATES),
        "per_model_set_seconds": per_set,
        "parts_seconds": {
            "tuning_gpu": total_gpu,
            "tuning_chrf_cpu": total_cpu,
            "final_decode_gpu_upper_bound": worst,
            "final_decode_chrf_cpu_upper_bound": worst_cpu,
            "bench": bench,
            "fixed_overhead": ASSUMED_FIXED_SECONDS,
        },
        "seconds": seconds,
        "hours": seconds / 3600,
        "cu": seconds / 3600 * CU_PER_HOUR,
    }


def format_final_estimate(est: Mapping[str, Any]) -> list[str]:
    """Printable ESTIMATE lines (labelled as such, with the basis)."""
    parts = est["parts_seconds"]
    return [
        f"ESTIMATE ({est['basis']}) -- not a measurement of this run:",
        f"  single-model rates: greedy {est['rates']['greedy']:.0f} out-tok/s, "
        f"beam {est['rates']['beam']:.0f} out-tok/s; {est['n_candidates']} candidates",
        "  parts (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in parts.items()),
        f"  total ~ {est['seconds'] / 60:.0f} min = {est['hours']:.1f} h "
        f"~ {est['cu']:.1f} CU at {CU_PER_HOUR} CU/h",
    ]


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "final_all steps")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tune", help="tune one of the 35 candidates (full E1+E2)")
    s.add_argument("--name", required=True, choices=FINAL_ALL_CANDIDATES)
    s.add_argument("--model", action="append", required=True, help="main|A|B=DIR (export dir)")
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--batch-size", type=int, default=ev.EVAL_BATCH_SIZE)

    s = sub.add_parser("select", help="selection.json over the 35 tuned candidates")
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)

    s = sub.add_parser("decode", help="decode every split (+test) for the winner")
    s.add_argument("--eval-dir", required=True, type=Path)
    s.add_argument("--model", action="append", required=True)
    s.add_argument("--batch-size", type=int, default=ev.EVAL_BATCH_SIZE)
    s.add_argument("--test", action="store_true")

    s = sub.add_parser("plan", help="print the ordered step list")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--repo", required=True)
    s.add_argument("--model", action="append", required=True)

    s = sub.add_parser("estimate", help="print the cost ESTIMATE of the exhaustive set")
    s.add_argument("--workload", type=Path, default=ev.REPO_ROOT / "colab" / "eval_workload.json")
    s.add_argument("--bench", type=Path, default=None, help="bench.json: use its measured rates")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except ev.EvalStepError as exc:
        print(f"final_all {args.cmd}: FAILED: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "tune":
        tune_candidate(args.name, parse_model_args(args.model), args.tuning_dir, args.batch_size)
    elif args.cmd == "select":
        if final_selection_is_valid(args.out, args.tuning_dir):
            print(f"select: SKIP (valid {args.out})")
        else:
            run_final_select(args.tuning_dir, args.out)
    elif args.cmd == "decode":
        decode_winner(
            args.eval_dir, parse_model_args(args.model), args.batch_size, include_test=args.test
        )
    elif args.cmd == "plan":
        models = parse_model_args(args.model)
        for name, argv in plan_final_all(
            eval_root=args.eval_root, hf_repo=args.repo, models=models
        ):
            print(f"{name}: {' '.join(argv)}")
    elif args.cmd == "estimate":
        workload = json.loads(args.workload.read_text(encoding="utf-8"))
        rates, basis = None, "ASSUMED rates (no bench yet)"
        bench = ev._read_json(args.bench) if args.bench else None
        if bench:
            m = bench["modes"]
            rates = {
                "greedy": float(m["greedy"]["output_tokens_per_second"]),
                "beam": float(m["beam5"]["output_tokens_per_second"]),
            }
            basis = f"MEASURED rates from {args.bench}"
        print("\n".join(format_final_estimate(estimate_final_all(workload, rates, basis))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
