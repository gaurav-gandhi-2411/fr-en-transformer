from __future__ import annotations

# Local, score-only evaluation pipeline for the Colab `eval_l4` outputs (RUNBOOK §4.5, §4.6).
# Pulls ONE or more runs' artifacts from the PRIVATE HF results repo at a pinned revision
# (a 40-hex commit sha, never a branch), then scores them:
#   - the official scorer exactly as shipped (official/score.py CLI), with an in-process parity
#     check,
#   - sacreBLEU, bootstrap 95% CIs (1000 resamples, seed 1234) per slice / E-set / length bucket /
#     E2-synth char bucket (nmt.evaluate.score_split_predictions, the code run_evaluation uses),
#   - the pre-registered paired tests (PREREG §3: H1 = S2 vs S1, H2 = S3 vs S2, primary at the
#     segmentation-off variant, secondary at the full tuned config) via nmt.compare with the
#     selection-objective bootstrap, whenever both runs are available on disk,
#   - COMET-22 through envs/comet (GPU only if one is free, else CPU; the choice is recorded),
#   - nmt.analysis (gap decomposition, length/rarity buckets, E2-synth, failure-mode rates, the
#     reference-normalization artifact delta).
#
# This script NEVER decodes and NEVER selects: decoding, tuning and selection happened on Colab
# (nmt.eval_l4), and everything here reads the uploaded predictions. `tests/test_eval_local.py`
# enforces that structurally (no Translator/tune/selection use) and behaviourally (any call to
# `Translator.translate`/`from_pretrained` fails the test).
#
# Outputs, under reports/eval/<run>/ (and reports/eval/compare/):
#   source/runs/<run>/...                 verbatim HF download (the revision it came from is pinned)
#   <variant>/<split>_predictions.json    the predictions scored (variant: seg_off, seg_tuned)
#   <variant>/eval.json  official_<split>.json  comet.json  analysis.json  examples.json  figures/
#   index.json                            every artifact + sha256 + source HF repo/revision
#   reports/eval/compare/H{1,2}_*.json + index.json
#
# CLI: `uv run python -m scripts.eval_local --hf-repo OWNER/NAME --revision <40-hex> --run main
#   [--run s1_sin_l4 --run s2_rope_l4 ...] [--run NAME@<40-hex>] [--out-root reports/eval]
#   [--n-bootstrap 1000] [--seed 1234] [--comet auto|cpu|gpu|off] [--no-analysis]`
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

from nmt.analysis import run_analysis
from nmt.compare import compare_eval_dirs
from nmt.eval_l4 import MANIFEST_NAME, expected_candidates, required_rels
from nmt.evaluate import (
    OFFICIAL_SCORE_PY,
    length_bucket_view,
    load_split,
    run_comet,
    run_official_scorer_cli,
    score_split_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "reports" / "eval"

RUNS = ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
SPLITS = ("dev", "e1", "e2", "e2synth", "e3")
VARIANTS = ("seg_off", "seg_tuned")
PRIMARY_VARIANT = "seg_off"  # PREREG §3: H1/H2 are decided with segmentation off
# (hypothesis id, run A, run B): Delta = A - B, PREREG §3.
HYPOTHESES = (("H1", "s2_rope_l4", "s1_sin_l4"), ("H2", "s3_rope_concat_l4", "s2_rope_l4"))
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# COMET-22 (xlm-roberta-large, batch 8) needs a few GiB; "free" means that much memory is
# unclaimed AND the GPU is not busy with another job (the 3070 is shared with intent-router).
COMET_MIN_FREE_MIB = 6144
COMET_MAX_UTIL_PCT = 25
COMET_TIMEOUT_SECONDS = 4 * 3600.0  # CPU COMET over ~4.4k triples is slow, never silent-fail early


class ScoreError(RuntimeError):
    """The artifacts are not what the pipeline needs (missing/invalid/unpinned): fail loudly."""


class ManifestVerificationError(ScoreError):
    """The pulled run is not what the uploaded manifest says (or the repo/revision is not the
    trusted one). Names the repo, revision, file and expected vs actual so it is actionable."""

    def __init__(
        self,
        reason: str,
        *,
        repo: str,
        revision: str,
        file: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        parts = [f"{repo}@{revision}: {reason}"]
        if file is not None:
            parts.append(f"file={file}")
        if expected is not None or actual is not None:
            parts.append(f"expected={expected!r} actual={actual!r}")
        super().__init__("; ".join(parts))
        self.repo, self.revision, self.file = repo, revision, file
        self.expected, self.actual = expected, actual


class Hub(Protocol):
    """The HF operations used here; `HubClient` is the real one, tests pass a fake."""

    def is_private(self, repo_id: str) -> bool | None: ...

    def list_files(self, repo_id: str, revision: str) -> list[str]: ...

    def download(self, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path: ...


class HubClient:
    """huggingface_hub-backed `Hub`. Read-only: it never creates, uploads or deletes anything."""

    def __init__(self) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi()  # HF_TOKEN / the cached login is used for the private repo

    def is_private(self, repo_id: str) -> bool | None:
        return getattr(self._api.repo_info(repo_id, repo_type="model"), "private", None)

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        return list(self._api.list_repo_files(repo_id, revision=revision, repo_type="model"))

    def download(self, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
            local_dir=str(local_dir),
            force_download=True,  # a cached copy is never trusted; bytes are re-verified anyway
        )
        return Path(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(text.encode("utf-8"))  # bytes: LF on every OS


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_run_spec(spec: str, default_revision: str | None) -> tuple[str, str]:
    """`NAME` or `NAME@<40-hex>` -> (run, revision). The revision must be a full commit sha:
    a branch name or short sha is not a pin, so it is refused."""
    name, _, rev = spec.partition("@")
    revision = rev or default_revision
    if name not in RUNS:
        raise ScoreError(f"--run {name!r} is not one of {RUNS}")
    if not revision or not _REVISION_RE.match(revision):
        raise ScoreError(
            f"run {name!r}: revision {revision!r} is not a 40-hex commit sha; pass --revision "
            "(or NAME@sha) with the HF_EVAL_REVISION the notebook printed"
        )
    return name, revision


# ---------------------------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------------------------


def required_files(run: str) -> list[str]:
    """Paths (relative to runs/<run>/) the scoring needs."""
    rels = ["selection.json", "decode_summary.json", "bench.json"]
    rels += [f"predictions/{v}/{s}_predictions.json" for v in VARIANTS for s in SPLITS]
    if run == "main":
        rels += ["test_predictions.json", "validation.json"]
    return rels


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PULL_RECORD_NAME = "pull_record.json"


def _unsafe_rel(rel: Any) -> str | None:
    """Why `rel` is not a safe path relative to runs/<run>/ (build_manifest's key form), or None.
    Refuses absolute (/x), colon-, NUL- and backslash-containing, empty and '..'-containing keys
    so a manifest can never steer a read/hash outside runs/<run>/."""
    if not isinstance(rel, str) or not rel:
        return "is not a non-empty string"
    if "\\" in rel:
        return "contains a backslash"
    if "\x00" in rel:
        return "contains a NUL character"
    if ":" in rel:
        # also closes NTFS alternate-data-stream names (a/b:stream) and drive-letter forms
        return "contains a colon"
    if rel.startswith("/"):
        return "is absolute"
    if any(part in ("..", ".", "") for part in rel.split("/")):
        return "has an empty, '.' or '..' path component"
    return None


def _load_manifest(path: Path, run: str, repo_id: str, revision: str) -> dict[str, Any]:
    """Parse and structurally validate runs/<run>/manifest.json (schema of nmt.eval_l4)."""

    def bad(reason: str, **kw: Any) -> ManifestVerificationError:
        return ManifestVerificationError(
            reason, repo=repo_id, revision=revision, file=f"runs/{run}/{MANIFEST_NAME}", **kw
        )

    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise bad(f"manifest is unreadable ({type(exc).__name__}: {exc})") from exc
    if not isinstance(manifest, dict):
        raise bad("manifest is not a JSON object")
    if manifest.get("run") != run:
        raise bad("manifest names a different run", expected=run, actual=manifest.get("run"))
    expected_order = list(expected_candidates(run))
    if manifest.get("candidate_order") != expected_order:
        raise bad(
            "manifest candidate_order is not the pre-registered one",
            expected=expected_order,
            actual=manifest.get("candidate_order"),
        )
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise bad("manifest has no file list")
    for rel, entry in listed.items():
        problem = _unsafe_rel(rel)
        if problem:
            raise bad(f"manifest key {rel!r} {problem}")
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("bytes"), int)
            and not isinstance(entry.get("bytes"), bool)
            and isinstance(entry.get("sha256"), str)
            and _SHA256_RE.match(entry["sha256"])
        )
        if not ok:
            raise bad(f"manifest entry for {rel!r} lacks a valid sha256/bytes")
    absent = [r for r in required_rels(run, expected_order) if r not in listed]
    if absent:
        raise bad("manifest lacks required file(s)", expected=sorted(absent), actual="absent")
    return manifest


def verify_pulled_run(
    repo_id: str,
    run: str,
    revision: str,
    source: Path,
    remote_files: list[str],
    with_model: bool = False,
) -> dict[str, Any]:
    """Check the bytes now on disk under <source>/runs/<run>/ against the HF manifest: every
    manifest file (except the model weights unless `with_model`) is present with the manifest's
    size and sha256 (computed here from the local bytes), and no local or remote file is absent
    from the manifest. Raises ManifestVerificationError; returns the verification record."""
    prefix = f"runs/{run}/"
    run_root = source / "runs" / run
    manifest_path = run_root / MANIFEST_NAME
    manifest = _load_manifest(manifest_path, run, repo_id, revision)
    listed: dict[str, dict[str, Any]] = manifest["files"]

    def fail(reason: str, **kw: Any) -> ManifestVerificationError:
        return ManifestVerificationError(reason, repo=repo_id, revision=revision, **kw)

    # Per-run scope: only hub files under runs/<run>/ are compared with the manifest. Anything
    # outside it (the root _write_probe.txt that nmt.eval_l4's write probe commits, .gitattributes,
    # README.md, other runs' runs/<other>/...) is not this run's and must neither refuse nor load.
    remote = {f.removeprefix(prefix) for f in remote_files if f.startswith(prefix)}
    remote.discard(MANIFEST_NAME)
    for rel in sorted(remote - set(listed)):
        raise fail("file on the hub is not in the manifest", file=prefix + rel)
    for rel in sorted(set(listed) - remote):
        raise fail("manifest lists a file absent from the hub", file=prefix + rel)

    def wanted(rel: str) -> bool:
        return with_model or not rel.startswith("model/")

    local = {p.relative_to(run_root).as_posix() for p in run_root.rglob("*") if p.is_file()}
    for rel in sorted(r for r in local - set(listed) - {MANIFEST_NAME} if wanted(r)):
        raise fail("local file is not in the manifest", file=prefix + rel)
    files: dict[str, dict[str, Any]] = {}
    for rel in sorted(listed):
        if not wanted(rel):
            continue
        want, path = listed[rel], run_root / rel
        if not path.is_file():
            raise fail("manifest file is missing locally", file=prefix + rel, expected="present")
        size = path.stat().st_size
        if size != want["bytes"]:
            raise fail(
                "size differs from the manifest",
                file=prefix + rel,
                expected=want["bytes"],
                actual=size,
            )
        digest = _sha256(path)
        if digest != want["sha256"]:
            raise fail(
                "sha256 differs from the manifest",
                file=prefix + rel,
                expected=want["sha256"],
                actual=digest,
            )
        files[rel] = {"ok": True, "sha256": digest, "bytes": size}
    return {
        "verified": True,
        "hf_repo": repo_id,
        "hf_revision": revision,
        "private": True,
        "manifest_sha256": _sha256(manifest_path),
        "model_files_checked": with_model,
        "files": files,
    }


def pull_run(
    hub: Hub, repo_id: str, run: str, revision: str, run_dir: Path, with_model: bool = False
) -> dict[str, str]:
    """Download runs/<run>/** at `revision` into <run_dir>/source/ and VERIFY it against the
    uploaded manifest.json (`verify_pulled_run`); any mismatch raises ManifestVerificationError
    before anything is scored. The repo must be private and the revision a 40-hex sha. The
    model weights are skipped unless `with_model`. Every call re-downloads and re-hashes: a
    cached pull is never trusted. Writes <run_dir>/pull_record.json. Returns {repo path: sha256}."""
    # a failed re-pull must not leave a previous run's 'verified' record behind
    (run_dir / PULL_RECORD_NAME).unlink(missing_ok=True)
    if not _REVISION_RE.match(revision or ""):
        raise ManifestVerificationError(
            "revision is not a 40-hex commit sha", repo=repo_id, revision=str(revision)
        )
    try:
        private = hub.is_private(repo_id)
    except Exception as exc:  # fail closed: an unreadable repo is not a trusted repo
        raise ManifestVerificationError(
            f"cannot read the repo's visibility ({type(exc).__name__}: {exc})",
            repo=repo_id,
            revision=revision,
        ) from exc
    if private is not True:
        raise ManifestVerificationError(
            "the HF results repo is not private",
            repo=repo_id,
            revision=revision,
            expected=True,
            actual=private,
        )
    # the hub lists the WHOLE repo; keep only runs/<run>/ so nothing else is ever downloaded
    prefix = f"runs/{run}/"
    files = [f for f in hub.list_files(repo_id, revision) if f.startswith(prefix)]
    present = {f.removeprefix(prefix) for f in files}
    missing = [r for r in required_files(run) if r not in present]
    if missing:
        raise ScoreError(f"{repo_id}@{revision}: runs/{run}/ is missing {missing}")
    if MANIFEST_NAME not in present:
        raise ManifestVerificationError(
            "manifest.json is missing from the run",
            repo=repo_id,
            revision=revision,
            file=prefix + MANIFEST_NAME,
        )
    source = run_dir / "source"
    hub.download(repo_id, prefix + MANIFEST_NAME, revision, source)
    _load_manifest(source / prefix / MANIFEST_NAME, run, repo_id, revision)  # fail before bulk
    digests: dict[str, str] = {}
    for f in sorted(files):
        if f.startswith(prefix + "model/") and not with_model:
            continue
        local = hub.download(repo_id, f, revision, source)
        digests[f] = _sha256(Path(local))
    record = verify_pulled_run(repo_id, run, revision, source, files, with_model)
    _write_json(run_dir / PULL_RECORD_NAME, record)
    return digests


# ---------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------


def load_predictions(path: Path, split: str) -> dict[str, str]:
    """The predictions of `split`, refusing a file whose ids are not exactly the split's ids."""
    pred = _read_json(path)
    ids = [r["id"] for r in load_split(split)[0]]
    if not isinstance(pred, dict) or set(pred) != set(ids):
        raise ScoreError(f"{path}: ids do not match the {split} split exactly")
    bad = [i for i, v in pred.items() if not isinstance(v, str) or not v.strip()]
    if bad:
        raise ScoreError(f"{path}: {len(bad)} empty/non-string prediction(s), e.g. {bad[:3]}")
    return pred


def gold_path(split: str) -> Path:
    """The gold labels file of `split` exactly as official/score.py's --gold expects it."""
    return REPO_ROOT / "data" / ("dev" if split == "dev" else f"eval/{split}") / "labels.jsonl"


def official_cli_parity(split: str, pred_path: Path, out_dir: Path, entry: dict[str, Any]) -> dict:
    """Run official/score.py as shipped on `split` and compare with the in-process numbers."""
    cli_path = out_dir / f"official_{split}.json"
    # run_official_scorer sets PYTHONUTF8=1, so the UTF-8 prediction file is read as UTF-8 on
    # every OS (official/score.py itself is byte-pinned and opens files with the default encoding).
    cli = run_official_scorer_cli(gold_path(split), pred_path, cli_path)
    mine = entry["official"]
    parity = (
        cli["all"] == mine["all"]
        and abs(cli["OVERALL"] - mine["OVERALL"]) < 1e-9
        and cli["by_slice"] == mine["by_slice"]
    )
    if not parity:
        raise ScoreError(f"{split}: official CLI report differs from the in-process scorer")
    return {"report": cli_path.name, "parity": True, "scorer_sha256": _sha256(OFFICIAL_SCORE_PY)}


def nvidia_smi_query() -> str | None:
    """`nvidia-smi` free-memory/utilisation of GPU 0 as 'free_mib, util_pct', None if absent."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def choose_comet_device(query: Any = nvidia_smi_query, mode: str = "auto") -> tuple[int, str]:
    """(gpus, reason): 1 only when a GPU is free (>= COMET_MIN_FREE_MIB free and <= 25% busy),
    else 0 (CPU). `mode` 'cpu'/'gpu' forces the choice. The reason is recorded in the output."""
    if mode == "cpu":
        return 0, "forced CPU (--comet cpu)"
    if mode == "gpu":
        return 1, "forced GPU (--comet gpu)"
    raw = query()
    if raw is None:
        return 0, "CPU: nvidia-smi unavailable or failed"
    try:
        free_mib, util = (float(x) for x in raw.split(","))
    except ValueError:
        return 0, f"CPU: could not parse nvidia-smi output {raw!r}"
    if free_mib >= COMET_MIN_FREE_MIB and util <= COMET_MAX_UTIL_PCT:
        return 1, f"GPU free: {free_mib:.0f} MiB free, {util:.0f}% utilisation"
    return 0, (
        f"CPU: GPU not free ({free_mib:.0f} MiB free < {COMET_MIN_FREE_MIB} or "
        f"{util:.0f}% > {COMET_MAX_UTIL_PCT}% utilisation)"
    )


def score_comet(
    triples_by_split: dict[str, list[dict[str, str]]], out_path: Path, mode: str, query: Any
) -> dict[str, Any]:
    """COMET-22 over every split's (src, mt, ref) triples; records the device used (and a GPU ->
    CPU fallback if the GPU run failed) plus the per-split system means."""
    gpus, reason = choose_comet_device(query, mode)
    triples = [t for s in SPLITS for t in triples_by_split[s]]
    fallback = None
    try:
        result = run_comet(triples, out_path, COMET_TIMEOUT_SECONDS, gpus=gpus)
    except RuntimeError as exc:
        if gpus == 0 or mode == "gpu":
            raise
        fallback = f"GPU run failed ({str(exc)[:200]}); retried on CPU"
        gpus = 0
        result = run_comet(triples, out_path, COMET_TIMEOUT_SECONDS, gpus=0)
    scores, start, by_split = result["scores"], 0, {}
    for s in SPLITS:
        n = len(triples_by_split[s])
        by_split[s] = sum(scores[start : start + n]) / n
        start += n
    result["system_score_by_split"] = by_split
    result["device"] = {"gpus": gpus, "reason": reason, "fallback": fallback}
    return result


def score_variant(
    run: str,
    variant: str,
    source_run_dir: Path,
    out_dir: Path,
    selection: dict[str, Any],
    *,
    provenance: dict[str, Any],
    n_bootstrap: int,
    seed: int,
    comet: str,
    comet_query: Any = nvidia_smi_query,
    do_analysis: bool = True,
) -> dict[str, Any]:
    """Score every split of one variant and write <out_dir>/{eval.json, official_*.json,
    comet.json, analysis.json, ...}. Reads only the uploaded predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    win = selection["winner"]
    threshold = None if variant == "seg_off" else win["segment_threshold"]
    result: dict[str, Any] = {
        "run": run,
        "checkpoint": win["candidate"],
        "variant": variant,
        "decoding_config": {
            "beam_size": win["beam"],
            "alpha": win["alpha"],
            "segment_threshold": threshold,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": seed,
        },
        "provenance": provenance,
        "sets": {},
    }
    all_pred: dict[str, str] = {}
    preds: dict[str, dict[str, str]] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    triples_by_split: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        src_pred = source_run_dir / "predictions" / variant / f"{split}_predictions.json"
        pred = preds[split] = load_predictions(src_pred, split)
        pred_path = out_dir / f"{split}_predictions.json"
        shutil.copyfile(src_pred, pred_path)
        inputs, labels = load_split(split)
        entry = score_split_predictions(split, inputs, labels, pred, n_bootstrap, seed)
        entry["official_cli"] = official_cli_parity(split, pred_path, out_dir, entry)
        result["sets"][split] = entry
        ref_by_id = {r["id"]: r["reference"] for r in labels}
        all_pred.update(pred)
        rows_by_split[split] = [
            {"id": r["id"], "source": r["source"], "reference": ref_by_id[r["id"]]} for r in inputs
        ]
        triples_by_split[split] = [
            {"src": r["source"], "mt": pred[r["id"]], "ref": ref_by_id[r["id"]]} for r in inputs
        ]
    combined = [row for s in ("e1", "e2", "e3") for row in rows_by_split[s]]
    result["length_buckets_e1_e2_e3"] = length_bucket_view(combined, all_pred, n_bootstrap, seed)
    if comet != "off":
        result["comet"] = score_comet(triples_by_split, out_dir / "comet.json", comet, comet_query)
        result["comet"].pop("scores", None)  # per-sentence scores stay in comet.json only
    _write_json(out_dir / "eval.json", result)
    if do_analysis:
        run_analysis(preds["e1"], preds["e3"], preds["dev"], out_dir, {f"{run}:{variant}": result})
    return result


# ---------------------------------------------------------------------------------------------
# paired tests
# ---------------------------------------------------------------------------------------------


def _chrf(entry: dict[str, Any]) -> dict[str, float]:
    return entry["overall"]["chrf"]


def prereg_readout(hyp: str, comparison: dict[str, Any]) -> dict[str, Any]:
    """A mechanical readout of PREREG §3's decision rule from one nmt.compare result (Delta = A-B,
    chrF). It restates the criteria and their values; it is not a substitute for reading them."""
    splits = comparison["splits"]

    def sig(split: str) -> bool:
        c = _chrf(splits[split])
        return c["delta"] > 0 and c["p_value"] < 0.05

    criteria: dict[str, bool] = {
        "e2: delta chrF > 0 and p < 0.05": sig("e2"),
        "e2synth (pooled): delta chrF > 0 and p < 0.05": sig("e2synth"),
    }
    if hyp == "H2":
        criteria["e1 non-inferiority: CI lower bound of delta chrF > -0.5"] = (
            _chrf(splits["e1"])["ci_low"] > -0.5
        )
    return {"criteria": criteria, "supported": all(criteria.values())}


def run_paired_tests(
    out_root: Path, n_bootstrap: int, seed: int, available: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """H1/H2 for every pair whose two runs both have complete variant dirs under `out_root`."""
    results: list[dict[str, Any]] = []
    for hyp, run_a, run_b in HYPOTHESES:
        if run_a not in available or run_b not in available:
            continue
        for variant in VARIANTS:
            comparison = compare_eval_dirs(
                out_root / run_a / variant,
                out_root / run_b / variant,
                ("dev", "e1", "e2", "e3"),
                n_bootstrap,
                seed,
                objective=True,
            )
            comparison.update(
                {
                    "hypothesis": hyp,
                    "delta": f"{run_a} - {run_b}",
                    "variant": variant,
                    "role": "primary" if variant == PRIMARY_VARIANT else "secondary",
                    "source": {
                        run_a: available[run_a],
                        run_b: available[run_b],
                    },
                    "prereg_readout": prereg_readout(hyp, comparison),
                }
            )
            path = out_root / "compare" / f"{hyp}_{run_a}_vs_{run_b}_{variant}.json"
            _write_json(path, comparison)
            results.append({"path": path, "hypothesis": hyp, "variant": variant})
    return results


# ---------------------------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------------------------


def _entries(root: Path, files: list[Path], origin: str, extra: dict[str, Any]) -> list[dict]:
    return [
        {
            "path": f.relative_to(root).as_posix(),
            "sha256": _sha256(f),
            "bytes": f.stat().st_size,
            "origin": origin,
            **extra,
        }
        for f in sorted(files)
    ]


def write_run_index(
    run_dir: Path, hf_repo: str, run: str, revision: str, digests: dict[str, str]
) -> dict[str, Any]:
    """index.json: every file of the run dir with its sha256 and where it came from. Downloaded
    files carry their HF path; derived files carry `derived_from_hf_revision`."""
    source = run_dir / "source"
    downloaded = [source / p for p in digests if (source / p).is_file()]
    derived = [
        f
        for f in run_dir.rglob("*")
        if f.is_file() and f.name != "index.json" and source not in f.parents
    ]
    artifacts = _entries(
        run_dir, downloaded, "hf_download", {"hf_repo": hf_repo, "hf_revision": revision}
    )
    for art in artifacts:
        art["hf_path"] = art["path"].removeprefix("source/")
    artifacts += _entries(
        run_dir, derived, "derived", {"hf_repo": hf_repo, "derived_from_hf_revision": revision}
    )
    index = {"run": run, "hf_repo": hf_repo, "hf_revision": revision, "artifacts": artifacts}
    _write_json(run_dir / "index.json", index)
    return index


def write_compare_index(out_root: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    cmp_dir = out_root / "compare"
    files = [r["path"] for r in results]
    _write_json(
        cmp_dir / "index.json",
        {"artifacts": _entries(cmp_dir, files, "derived", {})},
    )


# ---------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------


def evaluate_run(
    hub: Hub,
    hf_repo: str,
    run: str,
    revision: str,
    out_root: Path,
    *,
    n_bootstrap: int,
    seed: int,
    comet: str,
    comet_query: Any = nvidia_smi_query,
    do_analysis: bool = True,
) -> dict[str, Any]:
    """Pull one run at its pinned revision, score both variants, write its index.json."""
    run_dir = out_root / run
    digests = pull_run(hub, hf_repo, run, revision, run_dir)
    source_run_dir = run_dir / "source" / "runs" / run
    selection = _read_json(source_run_dir / "selection.json")
    provenance = {"hf_repo": hf_repo, "hf_revision": revision, "hf_path_prefix": f"runs/{run}/"}
    for variant in VARIANTS:
        score_variant(
            run,
            variant,
            source_run_dir,
            run_dir / variant,
            selection,
            provenance=provenance,
            n_bootstrap=n_bootstrap,
            seed=seed,
            comet=comet,
            comet_query=comet_query,
            do_analysis=do_analysis,
        )
    return write_run_index(run_dir, hf_repo, run, revision, digests)


def _complete_runs(out_root: Path) -> dict[str, dict[str, Any]]:
    """Runs with every variant's predictions + an index.json on disk: {run: {repo, revision}}."""
    found: dict[str, dict[str, Any]] = {}
    for run in RUNS:
        idx = out_root / run / "index.json"
        done = all(
            (out_root / run / v / f"{s}_predictions.json").is_file()
            for v in VARIANTS
            for s in SPLITS
        )
        if idx.is_file() and done:
            data = _read_json(idx)
            found[run] = {"hf_repo": data["hf_repo"], "hf_revision": data["hf_revision"]}
    return found


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score eval_l4 outputs pulled from the private HF repo."
    )
    p.add_argument("--hf-repo", required=True, help="OWNER/NAME of the private results repo")
    p.add_argument("--revision", default=None, help="40-hex commit sha (HF_EVAL_REVISION)")
    p.add_argument(
        "--run", action="append", required=True, help="NAME or NAME@<40-hex>; repeatable"
    )
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--comet", choices=("auto", "cpu", "gpu", "off"), default="auto")
    p.add_argument("--no-analysis", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, hub: Hub | None = None) -> int:
    args = _parse_args(argv)
    try:
        specs = [parse_run_spec(s, args.revision) for s in args.run]
        hub = hub or HubClient()
        for run, revision in specs:
            evaluate_run(
                hub,
                args.hf_repo,
                run,
                revision,
                args.out_root,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                comet=args.comet,
                do_analysis=not args.no_analysis,
            )
            print(f"eval_local: {run}@{revision} scored -> {args.out_root / run}")
        results = run_paired_tests(
            args.out_root, args.n_bootstrap, args.seed, _complete_runs(args.out_root)
        )
        write_compare_index(args.out_root, results)
        for r in results:
            print(f"eval_local: wrote {r['path']}")
    except ScoreError as exc:
        print(f"eval_local: FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
