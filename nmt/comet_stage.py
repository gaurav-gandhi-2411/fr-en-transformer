from __future__ import annotations

# The COMET-22 stage of the final_all Colab session. COMET runs on the L4 GPU (on CPU it would
# take hours). It runs after the final_all selection/decode/validation/upload and is non-fatal
# for them: the notebook prints COMET FAILED and exits non-zero, but never undoes them.
# Eval-only: nothing here feeds selection.
#
# What is scored (one COMET score per segment, per set; a "set" = system x variant x split):
#   final_all         seg_off, seg_tuned x dev, e1, e2, e2synth, e3 (the winner's decodes)
#   final_all_report  <candidate> x e1, e2 for the winner, runner-up and production config
#                     (report/predictions/<candidate>/, the report step's decodes)
#   main, s1_sin_l4, s2_rope_l4, s3_rope_concat_l4
#                     seg_off, seg_tuned x the 5 splits, PULLED from the private HF eval repo at
#                     PINNED revisions through scripts.eval_local.pull_run (manifest + sha256
#                     verified, private=True checked, 40-hex revision required)
#   copy_source       baseline: the source text copied as the output (id -> source) x the 5 splits
# Identical (src, hyp, ref) triples are scored once and the score is mapped back.
#
# Commands (each idempotent; the notebook runs them as subprocesses, never imports this module):
#   install  venv (system site packages: Colab's torch stays) + unbabel-comet, torch pinned
#   pull     the 4 existing runs from the private HF repo at the pinned revisions
#   score    dedup -> nmt.comet_worker (GPU) -> per-set JSON + comet_summary.json
#   upload   second private commit 'eval_l4: final_all COMET' under runs/final_all/comet/
#   plan     the COMET section of the dry run (sets, distinct triples, model, device, ESTIMATE)
# Output layout (<eval>/comet/): <system>/<variant>/<split>.json, comet_summary.json,
# comet_manifest.json (written by upload); scratch (never uploaded): _cache/ _pulled/ _inputs/.
import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from nmt import comet_worker as cw
from nmt import eval_l4 as ev

FINAL_ALL_RUN = ev.FINAL_ALL_RUN
COMET_DIRNAME = "comet"
SUMMARY_NAME = "comet_summary.json"
COMET_MANIFEST_NAME = "comet_manifest.json"
COMET_UPLOAD_RECORD = "comet_hf_upload.json"  # in <eval>/, next to hf_upload.json
COMET_COMMIT_MESSAGE = "eval_l4: final_all COMET"  # exact title; verify matches it
SCHEMA = 1
SPLITS: tuple[str, ...] = ev.DECODE_SPLITS
VARIANTS: tuple[str, ...] = ev.VARIANTS
REPORT_SPLITS: tuple[str, ...] = ("e1", "e2")
SYSTEM_FINAL_ALL = "final_all"
SYSTEM_REPORT = "final_all_report"
SYSTEM_BASELINE = "copy_source"
BASELINE_VARIANT = "baseline"
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 1234
HF_EVAL_REPO = "OWNER/fr-en-transformer-eval"
# The 4 existing runs and the HF eval repo revisions their reports were built from (equal to
# reports/final/<run>/pull_record.json, which a test checks). Never a branch name.
PINNED_RUNS: dict[str, str] = {
    "main": "c3d8598252853fcd7df1ef4a00e8b0382b8f4351",
    "s1_sin_l4": "041d49269f61d0cbd9c0380a4f0a0a31f7599547",
    "s2_rope_l4": "bdd850a6d384e088dd1eb52d67ad767149dee3af",
    "s3_rope_concat_l4": "ac92b8da971fdf64974089f40e8f4ddbf9d9638b",
}
COMET_VERSION = "2.2.7"  # unbabel-comet; the version the local CPU run and the pins were built on
MODEL_LICENSE = "apache-2.0 (Unbabel/wmt22-comet-da model card, LICENSE file)"
ENCODER_LICENSE = "mit (FacebookAI/xlm-roberta-large model card)"
# Colab-side venv (--system-site-packages keeps the preinstalled torch) and its constraints file.
DEFAULT_VENV = "/content/comet_venv"
# LOCAL runtime disk, never Drive: the pinned 2.3 GB checkpoint is linked/copied here, and
# Drive FUSE may refuse symlinks/hardlinks (the copy would then be 2.3 GB onto Drive).
DEFAULT_MODEL_DIR = "/content/comet_model"
DEFAULT_CONSTRAINTS = "/content/comet_constraints.txt"
# unbabel-comet 2.2.7 requires numpy<2 (no cp313 wheel -> `pip install unbabel-comet` cannot
# resolve on Python 3.13 without compiling numpy). On 3.13 the explicit dependency list below is
# installed with --no-deps and numpy stays at the installed 2.x (checked on CPU: scores equal the
# 3.12 path). Versions are the ones comet itself requires, minus numpy.
COMET_DEPS_NODEPS: tuple[str, ...] = (
    "entmax>=1.1,<2.0",
    "huggingface-hub>=0.19.3,<1.0",
    "jsonargparse==3.13.1",
    "pandas>=1.4.1",
    "protobuf>=4.24.4,<5.0.0",
    "pytorch-lightning>=2.0.0,<3.0.0",
    "sacrebleu>=2.0.0,<3.0.0",
    "scipy>=1.5.4,<2.0.0",
    "sentencepiece>=0.2.0,<0.3.0",
    "torchmetrics>=0.10.2,<0.11.0",
    "transformers>=4.17,<5.0",
)
# torchmetrics 0.10.x imports the legacy pkg_resources, which setuptools>=81 no longer ships.
SETUPTOOLS_REQ = "setuptools<81"

# ESTIMATE inputs. NO L4 COMET throughput has been measured: these are ASSUMED.
ASSUMED_COMET_RATES: tuple[float, ...] = (50.0, 100.0, 150.0)  # triples / s, fp32, batch 64
ASSUMED_COMET_CENTRAL = 100.0
ASSUMED_COMET_SETUP_SECONDS = 480.0  # pip install, 2.3 GB model download + load, 4 pulls
ASSUMED_COMET_UPLOAD_SECONDS = 120.0  # the second private commit (~2 MB of JSON)
ASSUMED_COMET_FIXED_SECONDS = ASSUMED_COMET_SETUP_SECONDS + ASSUMED_COMET_UPLOAD_SECONDS
FINAL_ALL_REPORT_CONFIGS = 3  # winner, runner-up, production (fewer when they coincide)


class CometStageError(RuntimeError):
    """A COMET-stage failure; the CLI prints it and exits 1 (the notebook prints COMET FAILED)."""


# ---------------------------------------------------------------------------------------------
# sets
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreSet:
    """One scored set: a prediction file of `system` / `variant` on `split`."""

    system: str
    variant: str
    split: str
    pred: Path

    @property
    def rel(self) -> str:
        """Output path relative to <eval>/comet/ (also the key in the comet manifest)."""
        return f"{self.system}/{self.variant}/{self.split}.json"


def comet_root(eval_root: Path) -> Path:
    return Path(eval_root) / COMET_DIRNAME


def pulled_predictions_dir(eval_root: Path, run: str) -> Path:
    """Where `pull` leaves a run's verified predictions (scripts.eval_local.pull_run layout)."""
    return comet_root(eval_root) / "_pulled" / run / "source" / "runs" / run / "predictions"


def baseline_path(eval_root: Path, split: str) -> Path:
    return comet_root(eval_root) / "_inputs" / "copy_source" / f"{split}_predictions.json"


def report_candidates(eval_root: Path) -> list[str]:
    """The distinct candidates of the report step, in winner / runner-up / production order."""
    report = ev._read_json(Path(eval_root) / "report.json")
    if not isinstance(report, dict):
        raise CometStageError(
            f"{eval_root}/report.json is missing: the final_all report step has not finished"
        )
    names: list[str] = []
    for key in ("winner", "runner_up", "production"):
        name = report.get(key)
        if not isinstance(name, str) or not name:
            raise CometStageError(f"{eval_root}/report.json has no {key!r} candidate")
        if name not in names:
            names.append(name)
    return names


def enumerate_sets(eval_root: Path, candidates: Sequence[str]) -> list[ScoreSet]:
    """EVERY set the stage scores, in a fixed order: the winner's 2 variants x 5 splits, the report
    predictions of each distinct candidate x (e1, e2), the 4 pulled runs x 2 variants x 5 splits,
    and the copy-source baseline x 5 splits."""
    root = Path(eval_root)
    sets = [
        ScoreSet(SYSTEM_FINAL_ALL, v, s, root / "predictions" / v / f"{s}_predictions.json")
        for v in VARIANTS
        for s in SPLITS
    ]
    sets += [
        ScoreSet(SYSTEM_REPORT, c, s, root / "report" / "predictions" / c / f"{s}_predictions.json")
        for c in candidates
        for s in REPORT_SPLITS
    ]
    sets += [
        ScoreSet(run, v, s, pulled_predictions_dir(root, run) / v / f"{s}_predictions.json")
        for run in PINNED_RUNS
        for v in VARIANTS
        for s in SPLITS
    ]
    sets += [ScoreSet(SYSTEM_BASELINE, BASELINE_VARIANT, s, baseline_path(root, s)) for s in SPLITS]
    return sets


def expected_set_count(n_candidates: int) -> int:
    """10 + 2 per report candidate + 4 x 10 + 5."""
    return (
        len(VARIANTS) * len(SPLITS)
        + len(REPORT_SPLITS) * n_candidates
        + (len(PINNED_RUNS) * len(VARIANTS) * len(SPLITS))
        + len(SPLITS)
    )


@cache
def load_split_data(split: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(ids, sources, references) of `split` in the split's inputs order."""
    from nmt.evaluate import load_split

    inputs, labels = load_split(split)
    ref_by_id = {r["id"]: r["reference"] for r in labels}
    ids = tuple(r["id"] for r in inputs)
    return ids, tuple(r["source"] for r in inputs), tuple(ref_by_id[i] for i in ids)


def copy_source_predictions(split: str) -> dict[str, str]:
    """The baseline: id -> the source text itself (deterministic, nothing else)."""
    ids, sources, _ = load_split_data(split)
    return dict(zip(ids, sources, strict=True))


def write_copy_source_baseline(eval_root: Path) -> list[Path]:
    """Write the baseline prediction file of every split (so its sha256 is recorded like any other
    input). Returns the paths."""
    paths = []
    for split in SPLITS:
        path = baseline_path(eval_root, split)
        ev._write_json(path, copy_source_predictions(split))
        paths.append(path)
    return paths


@dataclass(frozen=True)
class Prepared:
    """A set's inputs: sha256 of its prediction file and the (src, hyp, ref) triple per segment."""

    spec: ScoreSet
    ids: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]
    input_sha256: str
    empty_hypotheses: int


def prepare_set(spec: ScoreSet, pred: Mapping[str, Any] | None = None) -> Prepared:
    """Read one prediction file against its split; refuses a file whose ids are not exactly the
    split's ids or whose values are not strings (fail closed, never a guessed alignment). `pred`
    (the dry run's in-memory baseline) replaces the file read."""
    if pred is None:
        if not spec.pred.is_file():
            raise CometStageError(f"{spec.rel}: prediction file {spec.pred} is missing")
        pred = ev._read_json(spec.pred)
        sha = ev._sha256_file(spec.pred)
    else:
        sha = ""
    ids, sources, refs = load_split_data(spec.split)
    if not isinstance(pred, dict) or set(pred) != set(ids):
        raise CometStageError(f"{spec.pred}: ids do not match the {spec.split} split exactly")
    bad = [i for i in ids if not isinstance(pred[i], str)]
    if bad:
        raise CometStageError(f"{spec.pred}: {len(bad)} non-string prediction(s), e.g. {bad[:3]}")
    triples = tuple((s, pred[i], r) for i, s, r in zip(ids, sources, refs, strict=True))
    return Prepared(
        spec=spec,
        ids=ids,
        triples=triples,
        input_sha256=sha,
        empty_hypotheses=sum(1 for _, h, _ in triples if not h.strip()),
    )


def dedup_triples(
    prepared: Sequence[Prepared],
) -> tuple[list[tuple[str, str, str]], list[list[int]]]:
    """(distinct triples in first-occurrence order, per set the index of each segment's triple)."""
    position: dict[tuple[str, str, str], int] = {}
    unique: list[tuple[str, str, str]] = []
    layout: list[list[int]] = []
    for p in prepared:
        idx = []
        for t in p.triples:
            if t not in position:
                position[t] = len(unique)
                unique.append(t)
            idx.append(position[t])
        layout.append(idx)
    return unique, layout


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------


def bootstrap_mean_ci(
    scores: Sequence[float], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Mean and percentile 95% bootstrap CI over segments: `n_resamples` resamples with replacement,
    numpy default_rng(seed) index draws (the convention of nmt.evaluate.bootstrap_ci), the
    2.5th / 97.5th order statistics, clipped to bracket the point estimate."""
    n = len(scores)
    if n == 0:
        raise CometStageError("cannot bootstrap an empty score list")
    arr = np.asarray(scores, dtype=np.float64)
    mean = math.fsum(arr.tolist()) / n
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    means = np.sort(arr[idx].mean(axis=1))
    lo = min(float(means[int(0.025 * n_resamples)]), mean)
    hi = max(float(means[min(n_resamples - 1, int(0.975 * n_resamples))]), mean)
    return {"mean": mean, "ci95": [lo, hi], "n_resamples": n_resamples, "seed": seed}


# ---------------------------------------------------------------------------------------------
# per-set outputs
# ---------------------------------------------------------------------------------------------


def set_output_path(eval_root: Path, spec: ScoreSet) -> Path:
    return comet_root(eval_root) / spec.rel


def output_is_valid(path: Path, p: Prepared, precision: str) -> bool:
    """Resume check of one set: the file parses, is for this exact system/variant/split, was
    scored from THIS prediction file (sha256), at this model revision and precision, and holds one
    finite score per segment. Anything else is re-scored."""
    data = ev._read_json(path)
    if not isinstance(data, dict):
        return False
    scores = data.get("scores")
    return bool(
        data.get("schema") == SCHEMA
        and (data.get("system"), data.get("variant"), data.get("split"))
        == (p.spec.system, p.spec.variant, p.spec.split)
        and data.get("input_sha256") == p.input_sha256
        and data.get("ids") == list(p.ids)
        and (data.get("model") or {}).get("revision") == cw.COMET_MODEL_REVISION
        and data.get("precision") == precision
        and isinstance(scores, list)
        and len(scores) == len(p.ids)
        and all(
            isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s)
            for s in scores
        )
    )


def _model_record(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": cw.COMET_MODEL,
        "revision": cw.COMET_MODEL_REVISION,
        "license": MODEL_LICENSE,
        "encoder": cw.XLMR_REPO,
        "encoder_revision": cw.XLMR_REVISION,
        "encoder_license": ENCODER_LICENSE,
        "model_class": meta.get("model_class"),
        "class_identifier": meta.get("class_identifier"),
    }


def set_record(
    p: Prepared, scores: Sequence[float], meta: Mapping[str, Any], stage: Mapping[str, Any]
) -> dict[str, Any]:
    """The JSON of one set: per-segment scores (inputs order) + mean + bootstrap CI + provenance."""
    stats = bootstrap_mean_ci(scores)
    return {
        "schema": SCHEMA,
        "system": p.spec.system,
        "variant": p.spec.variant,
        "split": p.spec.split,
        "n": len(scores),
        "n_distinct_triples": len(set(p.triples)),
        "empty_hypotheses": p.empty_hypotheses,
        "ids": list(p.ids),
        "scores": [float(s) for s in scores],
        **stats,
        "ci_method": "percentile bootstrap over segments, numpy default_rng(seed).integers",
        "input_file": p.spec.pred.name,
        "input_sha256": p.input_sha256,
        "model": _model_record(meta),
        "libraries": meta.get("libraries"),
        "device": meta.get("device"),
        "precision": meta.get("precision"),
        "batch_size": meta.get("batch_size"),
        "runtime": dict(stage),
        "written_utc": datetime.now(UTC).isoformat(),
    }


def read_all_chunks(
    unique: Sequence[tuple[str, str, str]], cache_dir: Path, chunk_size: int
) -> tuple[list[float], float, dict[str, Any]]:
    """(scores of every distinct triple in order, summed chunk wall seconds). A missing or invalid
    chunk raises: scores are never partially assembled."""
    triples = [{"src": s, "mt": m, "ref": r} for s, m, r in unique]
    scores: list[float] = []
    seconds = 0.0
    meta: dict[str, Any] = {}
    for k, (lo, hi) in enumerate(cw.chunk_bounds(len(triples), chunk_size)):
        path = cw.chunk_path(cache_dir, k)
        part = cw.read_chunk(path, triples[lo:hi])
        if part is None:
            raise CometStageError(f"COMET chunk {k} ({path}) is missing or does not match")
        scores.extend(part)
        record = ev._read_json(path) or {}
        seconds += float(record.get("wall_seconds", 0.0))
        if not meta and isinstance(record.get("meta"), dict) and record["meta"].get("model"):
            meta = record["meta"]  # provenance of the first chunk that has it
    return scores, seconds, meta


# ---------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------


def venv_python(venv: Path) -> Path:
    """The interpreter of the COMET venv (POSIX bin/ or Windows Scripts/): whichever exists, else
    the platform's layout (the venv may not have been created yet when a command is built)."""
    posix, win = Path(venv) / "bin" / "python", Path(venv) / "Scripts" / "python.exe"
    if win.is_file() and not posix.is_file():
        return win
    if posix.is_file():
        return posix
    return win if sys.platform == "win32" else posix


def worker_command(
    python: Path | str,
    triples_path: Path,
    cache_dir: Path,
    meta_path: Path,
    *,
    model_dir: Path,
    chunk_size: int,
    batch_size: int,
    device: str,
    precision: str,
) -> list[str]:
    return [
        str(python),
        "-m",
        "nmt.comet_worker",
        "--in",
        str(triples_path),
        "--cache-dir",
        str(cache_dir),
        "--meta-out",
        str(meta_path),
        "--model-dir",
        str(model_dir),
        "--chunk-size",
        str(chunk_size),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--precision",
        precision,
    ]


# The worker downloads two PUBLIC models and needs no credential: it never sees the write token
# (least privilege), and a stale HUGGINGFACEHUB_API_TOKEN cannot make a public download fail.
WORKER_ENV_DROP = (
    "HF_TOKEN",
    "HF_TOKEN_WRITE",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GH_TOKEN",
    "WANDB_API_KEY",
)


def worker_env(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` minus every credential variable."""
    return {k: v for k, v in environ.items() if k not in WORKER_ENV_DROP}


def run_worker_subprocess(cmd: Sequence[str]) -> None:
    """Run the worker with its output passing straight through (the notebook streams and scrubs)."""
    done = subprocess.run(list(cmd), cwd=ev.REPO_ROOT, env=worker_env(os.environ), check=False)
    if done.returncode != 0:
        raise CometStageError(f"the COMET worker failed (exit {done.returncode}); see its output")


def check_model_dir_is_local(model_dir: Path, eval_root: Path) -> None:
    """Refuse a model dir under the eval root (Drive): the checkpoint link/copy is 2.3 GB."""
    model = Path(model_dir).resolve()
    root = Path(eval_root).resolve()
    if model == root or root in model.parents:
        raise CometStageError(
            f"COMET model dir {model} is under the eval dir {root} (Drive): it must be on "
            "the LOCAL runtime disk (e.g. /content/comet_model)"
        )


def score_stage(
    eval_root: Path,
    *,
    venv: Path = Path(DEFAULT_VENV),
    batch_size: int = cw.DEFAULT_BATCH_SIZE,
    precision: str = "fp32",
    device: str = "cuda",
    model_dir: Path = Path(DEFAULT_MODEL_DIR),
    chunk_size: int = cw.DEFAULT_CHUNK_SIZE,
    runner: Callable[[Sequence[str]], None] = run_worker_subprocess,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Score every set that has no valid output yet; (re)write comet_summary.json. `runner` runs
    the worker command (tests pass a stub). Returns the summary."""
    root = Path(eval_root)
    croot = comet_root(root)
    check_model_dir_is_local(model_dir, root)
    candidates = report_candidates(root)
    write_copy_source_baseline(root)
    specs = enumerate_sets(root, candidates)
    missing = [str(s.pred) for s in specs if not s.pred.is_file()]
    if missing:
        raise CometStageError(
            f"{len(missing)} prediction file(s) missing; the final_all decode, the report step "
            "and `pull` must have finished:\n  " + "\n  ".join(missing)
        )
    prepared = [prepare_set(s) for s in specs]
    pending = [
        p for p in prepared if not output_is_valid(set_output_path(root, p.spec), p, precision)
    ]
    log(
        f"comet: {len(prepared)} sets, {len(prepared) - len(pending)} already valid, "
        f"{len(pending)} to score"
    )
    # chunk scores depend on precision and model revision: never reuse them across either
    cache_dir = croot / "_cache" / f"{precision}-{cw.COMET_MODEL_REVISION[:8]}"
    meta_path = cache_dir / "worker_meta.json"
    if pending:
        unique, layout = dedup_triples(pending)
        total = sum(len(p.triples) for p in pending)
        log(f"comet: {len(unique)} distinct (src, hyp, ref) triples for {total} segments")
        triples_path = cache_dir / "triples.json"
        ev._write_json(triples_path, [{"src": s, "mt": m, "ref": r} for s, m, r in unique])
        py = venv_python(venv)
        runner(
            worker_command(
                py,
                triples_path,
                cache_dir,
                meta_path,
                model_dir=Path(model_dir),
                chunk_size=chunk_size,
                batch_size=batch_size,
                device=device,
                precision=precision,
            )
        )
        unique_scores, chunk_seconds, chunk_meta = read_all_chunks(unique, cache_dir, chunk_size)
        meta = ev._read_json(meta_path)
        if not isinstance(meta, dict) or not meta.get("model"):
            # killed between the last chunk and the meta write (or the meta was lost):
            # re-derive the record from the chunk cache's own provenance
            meta = chunk_meta
        if not meta.get("model"):
            raise CometStageError(f"{meta_path} and the chunk cache record no loaded model")
        stage = {
            "scoring_wall_seconds": chunk_seconds,
            "distinct_triples_scored": len(unique),
            "segments_covered": total,
            "dedup": "identical (src, hyp, ref) triples are scored once and share the score",
            "scope": "this stage run (all sets scored in it), not per set",
        }
        for p, idx in zip(pending, layout, strict=True):
            record = set_record(p, [unique_scores[i] for i in idx], meta, stage)
            ev._write_json(set_output_path(root, p.spec), record)
    return write_summary(root, prepared, precision)


def write_summary(eval_root: Path, prepared: Sequence[Prepared], precision: str) -> dict[str, Any]:
    """comet_summary.json from the per-set outputs on disk (every one is re-validated)."""
    root = Path(eval_root)
    rows, any_record = [], None
    for p in prepared:
        path = set_output_path(root, p.spec)
        if not output_is_valid(path, p, precision):
            raise CometStageError(f"{path} is missing or invalid after scoring")
        rec = ev._read_json(path) or {}
        any_record = rec
        rows.append(
            {
                "system": p.spec.system,
                "variant": p.spec.variant,
                "split": p.spec.split,
                "n": rec["n"],
                "mean": rec["mean"],
                "ci95": rec["ci95"],
                "n_distinct_triples": rec["n_distinct_triples"],
                "file": p.spec.rel,
                "input_file": rec["input_file"],
                "input_sha256": rec["input_sha256"],
                "output_sha256": ev._sha256_file(path),
            }
        )
    unique_all, _ = dedup_triples(prepared)
    pulls = {}
    for run, rev in PINNED_RUNS.items():
        rec = ev._read_json(comet_root(root) / "_pulled" / run / "pull_record.json") or {}
        pulls[run] = {
            "pinned_revision": rev,
            "verified": rec.get("verified"),
            "private": rec.get("private"),
            "manifest_sha256": rec.get("manifest_sha256"),
        }
    summary = {
        "schema": SCHEMA,
        "run": FINAL_ALL_RUN,
        "kind": "comet",
        "eval_only": "never used by selection or training",
        "model": (any_record or {}).get("model"),
        "libraries": (any_record or {}).get("libraries"),
        "device": (any_record or {}).get("device"),
        "precision": precision,
        "batch_size": (any_record or {}).get("batch_size"),
        "runtime": (any_record or {}).get("runtime"),
        "hf_eval_repo_pulls": pulls,
        "bootstrap": {"n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "n_sets": len(rows),
        "n_segments": sum(r["n"] for r in rows),
        "n_distinct_triples_all_sets": len(unique_all),
        "sets": rows,
        "written_utc": datetime.now(UTC).isoformat(),
    }
    ev._write_json(comet_root(root) / SUMMARY_NAME, summary)
    return summary


# ---------------------------------------------------------------------------------------------
# pull (the 4 existing runs, pinned + manifest-verified)
# ---------------------------------------------------------------------------------------------


def pull_runs(
    eval_root: Path, hf_repo: str, hub: Any | None = None, runs: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Pull every pinned run's predictions with scripts.eval_local.pull_run (private=True checked,
    40-hex revision, every file's size + sha256 compared with the uploaded manifest; every call
    re-downloads and re-hashes). Returns {run: pull_record}. Any mismatch raises."""
    from scripts import eval_local as el  # lazy: pulls in torch and the analysis stack

    hub = hub or el.HubClient()
    out = {}
    for run, revision in (runs or PINNED_RUNS).items():
        run_dir = comet_root(eval_root) / "_pulled" / run
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            el.pull_run(hub, hf_repo, run, revision, run_dir)
        except el.ScoreError as exc:
            raise CometStageError(f"pull {run}@{revision}: {exc}") from exc
        record = ev._read_json(run_dir / el.PULL_RECORD_NAME)
        if not (
            isinstance(record, dict)
            and record.get("verified") is True
            and record.get("private") is True
            and record.get("hf_revision") == revision
        ):
            raise CometStageError(f"pull {run}: the pull record is not verified/private/pinned")
        out[run] = record
        print(f"pull {run}@{revision}: verified against the manifest (private=True)")
    return out


# ---------------------------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------------------------


def installed_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError as exc:
        raise CometStageError(
            f"{package} is not installed in this interpreter ({sys.executable}); on Colab torch "
            "is preinstalled (check Runtime type)"
        ) from exc


def install_mode(mode: str, python_version: tuple[int, int]) -> str:
    """'resolve' (pip resolves unbabel-comet, numpy<2 lands in the venv) on Python <= 3.12;
    'nodeps' on 3.13+, where numpy<2 has no wheel. `mode` auto picks by version."""
    if mode not in ("auto", "resolve", "nodeps"):
        raise CometStageError(f"install mode {mode!r} is not auto, resolve or nodeps")
    if mode != "auto":
        return mode
    return "resolve" if python_version < (3, 13) else "nodeps"


def constraint_lines(mode: str, torch_version: str, numpy_version: str) -> list[str]:
    """torch pinned to the preinstalled build (pip may never swap it); numpy too in nodeps mode."""
    lines = [f"torch=={torch_version}"]
    if mode == "nodeps":
        lines.append(f"numpy=={numpy_version}")
    return lines


def install_commands(
    python: str, venv: Path, constraints: Path, mode: str
) -> list[tuple[str, list[str]]]:
    """(step name, argv) list: create the venv (system site packages, no ensurepip needed), then
    pip (run from the base interpreter, --python = the venv) with the constraints file and wheels
    only (no source builds)."""
    vpy = str(venv_python(venv))
    pip = [
        python,
        "-m",
        "pip",
        "--python",
        vpy,
        "install",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "-c",
        str(constraints),
    ]
    steps = [
        ("venv", [python, "-m", "venv", "--system-site-packages", "--without-pip", str(venv)]),
    ]
    if mode == "resolve":
        steps.append(("pip", [*pip, f"unbabel-comet=={COMET_VERSION}", SETUPTOOLS_REQ]))
    else:
        steps.append(
            ("pip comet --no-deps", [*pip, "--no-deps", f"unbabel-comet=={COMET_VERSION}"])
        )
        steps.append(("pip dependencies", [*pip, *COMET_DEPS_NODEPS, SETUPTOOLS_REQ]))
    return steps


PROBE_CODE = (
    "import importlib.metadata as m, json, comet; "
    "print(json.dumps({p: m.version(p) for p in ('unbabel-comet', 'torch', 'numpy')}))"
)


def probe_venv(venv: Path, run: Callable[..., Any] = subprocess.run) -> dict[str, str] | None:
    """Versions seen inside the venv, or None when it is missing or cannot import comet."""
    vpy = venv_python(venv)
    if not vpy.is_file():
        return None
    try:
        done = run([str(vpy), "-c", PROBE_CODE], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        return dict(json.loads(done.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return None


def install_comet(
    venv: Path,
    constraints: Path,
    mode: str = "auto",
    python: str | None = None,
    run: Callable[..., Any] = subprocess.run,
    python_version: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Install unbabel-comet into `venv` WITHOUT touching the preinstalled torch. Idempotent: a
    venv that already imports comet at the pinned version with the same torch is kept. A failure
    raises CometStageError naming the step and the Python version."""
    python = python or sys.executable
    pyv = python_version or (sys.version_info.major, sys.version_info.minor)
    chosen = install_mode(mode, pyv)
    torch_before = installed_version("torch")
    numpy_before = installed_version("numpy")
    probe = probe_venv(venv, run)
    if probe and probe.get("unbabel-comet") == COMET_VERSION and probe.get("torch") == torch_before:
        print(
            f"install: SKIP (venv {venv} already has unbabel-comet {COMET_VERSION}, "
            f"torch {torch_before})"
        )
        return {"mode": chosen, "venv": str(venv), "versions": probe, "skipped": True}
    constraints = Path(constraints)
    constraints.parent.mkdir(parents=True, exist_ok=True)
    constraints.write_text(
        "\n".join(constraint_lines(chosen, torch_before, numpy_before)) + "\n", encoding="utf-8"
    )
    print(
        f"install: python {pyv[0]}.{pyv[1]}, mode {chosen}, constraints: "
        + ", ".join(constraint_lines(chosen, torch_before, numpy_before))
    )
    if chosen == "nodeps":
        print(
            "install: UNVERIFIED ON COLAB: --no-deps path (Python 3.13: unbabel-comet needs "
            "numpy<2, which has no 3.13 wheel); checked only on a Windows CPU throwaway venv"
        )
    for name, argv in install_commands(python, Path(venv), constraints, chosen):
        print(f"install[{name}]: {' '.join(argv)}", flush=True)
        done = run(argv, check=False)
        if done.returncode != 0:
            raise CometStageError(
                f"COMET install failed at step '{name}' (exit {done.returncode}; python "
                f"{pyv[0]}.{pyv[1]}, mode {chosen}). The preinstalled torch {torch_before} was "
                "not modified (it is pinned by the constraints file and the install goes into "
                f"the separate venv {venv}). Read the pip output above; if python -m venv is "
                "unavailable install python3-venv or use a fresh runtime."
            )
    probe = probe_venv(venv, run)
    if not probe or probe.get("unbabel-comet") != COMET_VERSION:
        raise CometStageError(
            f"COMET install: the venv {venv} cannot import comet {COMET_VERSION} after pip "
            f"(probe: {probe})"
        )
    if probe.get("torch") != torch_before:
        raise CometStageError(
            f"COMET install changed torch: venv sees {probe.get('torch')}, the preinstalled "
            f"build is {torch_before}; delete the runtime and re-run"
        )
    print(f"install: OK {probe} (kernel torch {torch_before} untouched)")
    return {"mode": chosen, "venv": str(venv), "versions": probe, "skipped": False}


# ---------------------------------------------------------------------------------------------
# upload (second private commit under runs/final_all/comet/)
# ---------------------------------------------------------------------------------------------


def comet_upload_files(eval_root: Path) -> list[tuple[Path, str]]:
    """(local path, path in repo) of every file of the COMET upload: the per-set JSONs and
    comet_summary.json; scratch dirs (_cache, _pulled, _inputs) and the manifest itself excluded.
    Raises when the summary is missing or any set it lists is."""
    croot = comet_root(eval_root)
    summary = ev._read_json(croot / SUMMARY_NAME)
    if not isinstance(summary, dict) or not summary.get("sets"):
        raise CometStageError(f"{croot / SUMMARY_NAME} is missing: run the COMET score step first")
    rels = [SUMMARY_NAME, *sorted(s["file"] for s in summary["sets"])]
    missing = [str(croot / r) for r in rels if not (croot / r).is_file()]
    if missing:
        raise CometStageError("COMET upload: missing file(s): " + ", ".join(missing))
    prefix = f"runs/{FINAL_ALL_RUN}/{COMET_DIRNAME}/"
    return [(croot / r, prefix + r) for r in rels]


def build_comet_manifest(files: Sequence[tuple[Path, str]]) -> dict[str, Any]:
    """sha256 + size of every uploaded file, keys relative to runs/final_all/comet/."""
    prefix = f"runs/{FINAL_ALL_RUN}/{COMET_DIRNAME}/"
    return {
        "schema": SCHEMA,
        "run": FINAL_ALL_RUN,
        "kind": "comet",
        "model": cw.COMET_MODEL,
        "model_revision": cw.COMET_MODEL_REVISION,
        "files": {
            rel.removeprefix(prefix): {"sha256": ev._sha256_file(p), "bytes": p.stat().st_size}
            for p, rel in files
        },
    }


def verify_comet_on_hf(api: Any, repo_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Is exactly this COMET upload already on the private repo? True only when a commit titled
    COMET_COMMIT_MESSAGE exists and the manifest at the head equals `manifest` with every listed
    file present at the manifest's size and sha256. Unreadable evidence raises (fail closed)."""
    prefix = f"runs/{FINAL_ALL_RUN}/{COMET_DIRNAME}"
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        head = getattr(info, "sha", None)
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
        if ev._status(exc) == 404:
            return {"complete": False, "reason": f"nothing under {prefix}/", "revision": None}
        raise ev.EvalStepError(f"hf-verify COMET: cannot read {repo_id} ({exc})") from exc
    upload = next((c for c in commits if getattr(c, "title", None) == COMET_COMMIT_MESSAGE), None)
    if upload is None:
        return {"complete": False, "reason": "no COMET commit", "revision": None}
    remote = {e.path: e for e in entries if getattr(e, "size", None) is not None}

    def fetch(rel: str) -> Path:
        return Path(
            api.hf_hub_download(
                repo_id=repo_id, filename=f"{prefix}/{rel}", repo_type="model", revision=head
            )
        )

    if f"{prefix}/{COMET_MANIFEST_NAME}" not in remote:
        return {"complete": False, "reason": "COMET manifest missing", "revision": None}
    remote_manifest = ev._read_json(fetch(COMET_MANIFEST_NAME))
    if remote_manifest != dict(manifest):
        return {"complete": False, "reason": "remote COMET manifest differs", "revision": None}
    for rel, want in manifest["files"].items():
        entry = remote.get(f"{prefix}/{rel}")
        if entry is None or entry.size != want["bytes"]:
            return {"complete": False, "reason": f"{rel}: absent or wrong size", "revision": None}
        if (ev._lfs_sha256(entry) or ev._sha256_file(fetch(rel))) != want["sha256"]:
            return {"complete": False, "reason": f"{rel}: sha256 differs", "revision": None}
    return {
        "complete": True,
        "reason": "manifest and sha256 verified",
        "revision": upload.commit_id,
    }


def upload_comet(api: Any, repo_id: str, eval_root: Path) -> dict[str, Any]:
    """Second PRIVATE upload commit ('eval_l4: final_all COMET') under runs/final_all/comet/, with
    own comet_manifest.json (sha256 of every file). Refuses unless the MAIN final_all upload is
    complete on the repo (verify_run_on_hf: manifest + sha256, repo private); token scope checked;
    privacy asserted before and after the commit; skipped on HF evidence that exactly this upload
    exists. Records <eval>/comet_hf_upload.json and prints HF_COMET_REVISION=<sha>."""
    from huggingface_hub import CommitOperationAdd

    root = Path(eval_root)
    files = comet_upload_files(root)
    ev.check_write_token(api, repo_id)
    main_state = ev.verify_run_on_hf(api, repo_id, FINAL_ALL_RUN)  # also refuses a public repo
    if not main_state["complete"]:
        raise CometStageError(
            f"the main {FINAL_ALL_RUN} upload is not complete on {repo_id} "
            f"({main_state['reason']}); the COMET commit is only made after it"
        )
    manifest = build_comet_manifest(files)
    ev.assert_repo_private(api, repo_id, "before the COMET upload")
    state = verify_comet_on_hf(api, repo_id, manifest)
    if state["complete"]:
        ev.assert_repo_private(api, repo_id, "COMET upload already done")
        print(f"upload-comet: SKIP (verified on HF at revision {state['revision']})")
        revision, verified_existing = state["revision"], True
    else:
        manifest_path = comet_root(root) / COMET_MANIFEST_NAME
        ev._write_json(manifest_path, manifest)
        prefix = f"runs/{FINAL_ALL_RUN}/{COMET_DIRNAME}/"
        operations = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in files]
        operations.append(
            CommitOperationAdd(
                path_in_repo=prefix + COMET_MANIFEST_NAME, path_or_fileobj=str(manifest_path)
            )
        )
        try:
            commit = api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message=COMET_COMMIT_MESSAGE,
            )
        except Exception as exc:
            if ev._is_auth_error(exc):
                raise ev.HFWriteTokenError(
                    ev.write_token_message(repo_id, f"COMET upload refused: {exc}")
                ) from exc
            raise
        revision = getattr(commit, "oid", None) or getattr(commit, "commit_oid", None)
        ev.assert_repo_private(api, repo_id, "after the COMET upload")
        if not revision or not ev._REVISION_RE.match(str(revision)):
            raise ev.EvalStepError(f"COMET upload: no usable revision sha ({revision!r})")
        verified_existing = False
    record = {
        "repo": repo_id,
        "run": FINAL_ALL_RUN,
        "kind": "comet",
        "revision": revision,
        "private": True,
        "files": [r for _, r in files],
        "verified_existing": verified_existing,
    }
    ev._write_json(root / COMET_UPLOAD_RECORD, record)
    print(f"HF_COMET_REVISION={revision}")
    return record


# ---------------------------------------------------------------------------------------------
# dry-run section: sets, distinct triples, model, device, ESTIMATE
# ---------------------------------------------------------------------------------------------


def planned_counts(
    eval_root: Path, runs_root: Path | None, n_candidates: int | None = None
) -> dict[str, Any]:
    """Segment and distinct-triple counts of the stage. Exact where the files exist (the 4 runs'
    predictions under `runs_root` = reports/final of the checkout; the baseline, derived from the
    inputs; final_all from the eval dir), else an UNDEDUPED formula upper bound (`exact` False)."""
    seg = {s: len(load_split_data(s)[0]) for s in SPLITS}
    seg_all = sum(seg.values())
    n_c = FINAL_ALL_REPORT_CONFIGS if n_candidates is None else n_candidates
    formula = {
        "runs": len(PINNED_RUNS) * len(VARIANTS) * seg_all,
        "baseline": seg_all,
        "final_all": len(VARIANTS) * seg_all + n_c * sum(seg[s] for s in REPORT_SPLITS),
    }
    groups: dict[str, list[Prepared] | None] = {
        "baseline": [
            prepare_set(
                ScoreSet(SYSTEM_BASELINE, BASELINE_VARIANT, s, baseline_path(eval_root, s)),
                copy_source_predictions(s),
            )
            for s in SPLITS
        ]
    }
    run_specs = [
        ScoreSet(r, v, s, Path(runs_root or "") / r / v / f"{s}_predictions.json")
        for r in PINNED_RUNS
        for v in VARIANTS
        for s in SPLITS
    ]
    have_runs = runs_root is not None and all(x.pred.is_file() for x in run_specs)
    groups["runs"] = [prepare_set(x) for x in run_specs] if have_runs else None
    try:
        fa_specs = [
            x
            for x in enumerate_sets(eval_root, report_candidates(eval_root))
            if x.system in (SYSTEM_FINAL_ALL, SYSTEM_REPORT)
        ]
    except CometStageError:
        fa_specs = []
    have_fa = bool(fa_specs) and all(x.pred.is_file() for x in fa_specs)
    groups["final_all"] = [prepare_set(x) for x in fa_specs] if have_fa else None
    known = [p for g in groups.values() if g is not None for p in g]
    union = len(dedup_triples(known)[0])
    return {
        "segments": formula,
        "distinct_exact": {
            k: (len(dedup_triples(g)[0]) if g is not None else None) for k, g in groups.items()
        },
        "distinct_known_union": union,
        "distinct_upper_bound": union + sum(formula[k] for k, g in groups.items() if g is None),
        "exact": all(g is not None for g in groups.values()),
        "n_candidates": n_c,
    }


def estimate_comet(
    n_triples: int,
    rates: Sequence[float] = ASSUMED_COMET_RATES,
    fixed_seconds: float = ASSUMED_COMET_FIXED_SECONDS,
) -> dict[str, Any]:
    """ESTIMATE of the COMET stage on the L4: scoring time at each ASSUMED triples/s rate plus an
    ASSUMED fixed overhead (install, 2.3 GB model download + load, pulls, second upload). Nothing
    here was measured on a GPU."""
    rows = [
        {
            "triples_per_second": r,
            "scoring_seconds": n_triples / r,
            "total_seconds": n_triples / r + fixed_seconds,
            "total_hours": (n_triples / r + fixed_seconds) / 3600,
        }
        for r in rates
    ]
    return {"n_triples": n_triples, "fixed_seconds": fixed_seconds, "rows": rows}


def describe_stage(
    eval_root: Path,
    runs_root: Path | None,
    *,
    batch_size: int = cw.DEFAULT_BATCH_SIZE,
    precision: str = "fp32",
    device: str = "cuda",
    n_candidates: int | None = None,
) -> list[str]:
    """The COMET section of the dry run: what is scored, the distinct-triple counts (exact where
    the files exist here, else an upper bound), model, device, batch size and the ESTIMATE."""
    c = planned_counts(eval_root, runs_root, n_candidates)
    n_c = c["n_candidates"]
    seg = c["segments"]
    ex = c["distinct_exact"]

    def count(key: str) -> str:
        return (
            f"{ex[key]} distinct (exact, from the files here)"
            if ex[key] is not None
            else f"<= {seg[key]} (formula upper bound, no dedup; exact once the files exist)"
        )

    est = estimate_comet(c["distinct_upper_bound"])
    lines = [
        "COMET stage: after the upload, NON-FATAL for it, resumable per set "
        f"({expected_set_count(n_c)} sets with {n_c} report candidate(s); the report decodes are "
        "known only after the report step)",
        f"  model: {cw.COMET_MODEL} @ {cw.COMET_MODEL_REVISION} (unbabel-comet {COMET_VERSION}); "
        f"licence {MODEL_LICENSE}",
        f"  encoder files: {cw.XLMR_REPO} @ {cw.XLMR_REVISION}; licence {ENCODER_LICENSE}",
        f"  device: {device} ('cuda' refuses without CUDA; 'auto' falls back to the CPU "
        f"with a loud message; 'cpu' only when set explicitly); "
        f"batch size {batch_size}; precision {precision}; chunk size "
        f"{cw.DEFAULT_CHUNK_SIZE}; seed {cw.DEFAULT_SEED}",
        f"  bootstrap: {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, 95% percentile CI "
        "over segments (per set)",
        "  systems / variants / splits:",
        f"    {SYSTEM_FINAL_ALL}: {', '.join(VARIANTS)} x {', '.join(SPLITS)} "
        f"({len(VARIANTS) * len(SPLITS)} sets)",
        f"    {SYSTEM_REPORT}: winner / runner-up / production config x "
        f"{', '.join(REPORT_SPLITS)} (2 per distinct config, 3 configs at most)",
    ]
    for run, rev in PINNED_RUNS.items():
        lines.append(
            f"    {run}: {', '.join(VARIANTS)} x {', '.join(SPLITS)} (10 sets), pulled from "
            f"{HF_EVAL_REPO}@{rev} (manifest-verified, private)"
        )
    lines += [
        f"    {SYSTEM_BASELINE}: source text copied as the output x {', '.join(SPLITS)} "
        f"({len(SPLITS)} sets)",
        f"  segments before dedup: 4 runs {seg['runs']}, baseline {seg['baseline']}, "
        f"final_all (+report) {seg['final_all']}",
        f"  distinct (src, hyp, ref) triples: 4 runs {count('runs')}; baseline "
        f"{count('baseline')}; final_all {count('final_all')}",
        f"  distinct triples to score: {'exactly' if c['exact'] else 'at most'} "
        f"{c['distinct_upper_bound']}"
        + ("" if c["exact"] else " (known parts exact, the rest an upper bound)"),
        "  ESTIMATE (ASSUMED throughput, NO L4 COMET rate has been measured; fixed overhead "
        f"ASSUMED {est['fixed_seconds'] / 60:.0f} min for install + 2.3 GB model download/load + "
        "pulls + second upload):",
    ]
    for row in est["rows"]:
        lines.append(
            f"    ASSUMED {row['triples_per_second']:.0f} triples/s: scoring "
            f"{row['scoring_seconds'] / 60:.1f} min, stage total {row['total_seconds'] / 60:.1f} "
            f"min = {row['total_hours']:.2f} h"
        )
    return lines


# ---------------------------------------------------------------------------------------------
# plan steps + CLI
# ---------------------------------------------------------------------------------------------


def plan_steps(
    *,
    eval_root: Path,
    hf_repo: str,
    python: str | None = None,
    venv: Path = Path(DEFAULT_VENV),
    constraints: Path = Path(DEFAULT_CONSTRAINTS),
    batch_size: int = cw.DEFAULT_BATCH_SIZE,
    precision: str = "fp32",
    device: str = "cuda",
) -> list[tuple[str, list[str]]]:
    """The COMET steps appended to the final_all plan, after `upload`; each argv a fresh
    interpreter and each idempotent. Names start with `comet:` (the notebook runs them after the
    selection/upload steps and treats a failure as COMET FAILED, not as a failed run)."""
    py = python or sys.executable
    base = [py, "-m", "nmt.comet_stage"]
    install = [*base, "install", "--venv", str(venv), "--constraints", str(constraints)]
    steps: list[tuple[str, list[str]]] = [("comet:install", install)]
    for run in PINNED_RUNS:
        pull = [*base, "pull", "--eval-root", str(eval_root), "--hf-repo", hf_repo, "--run", run]
        steps.append((f"comet:pull:{run}", pull))
    score = [
        *base,
        "score",
        "--eval-root",
        str(eval_root),
        "--venv",
        str(venv),
        "--batch-size",
        str(batch_size),
        "--precision",
        precision,
        "--device",
        device,
    ]
    steps.append(("comet:score", score))
    upload = [*base, "upload", "--eval-root", str(eval_root), "--repo", hf_repo]
    steps.append(("comet:upload", upload))
    return steps


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "COMET stage")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("install", help="venv + unbabel-comet, torch pinned")
    s.add_argument("--venv", type=Path, default=Path(DEFAULT_VENV))
    s.add_argument("--constraints", type=Path, required=True)
    s.add_argument("--mode", choices=("auto", "resolve", "nodeps"), default="auto")

    s = sub.add_parser("pull", help="pull the pinned existing runs (manifest-verified)")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--hf-repo", default=HF_EVAL_REPO)
    s.add_argument("--run", action="append", choices=tuple(PINNED_RUNS), default=None)

    s = sub.add_parser("score", help="dedup, score on the GPU, per-set JSON + summary")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--venv", type=Path, default=Path(DEFAULT_VENV))
    s.add_argument("--batch-size", type=int, default=cw.DEFAULT_BATCH_SIZE)
    s.add_argument("--precision", choices=cw.PRECISIONS, default="fp32")
    s.add_argument("--device", choices=cw.DEVICES, default="cuda")
    s.add_argument("--chunk-size", type=int, default=cw.DEFAULT_CHUNK_SIZE)
    s.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))

    s = sub.add_parser("upload", help="second private commit under runs/final_all/comet/")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--repo", required=True)

    s = sub.add_parser("plan", help="print the COMET section of the dry run")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--runs-root", type=Path, default=ev.REPO_ROOT / "reports" / "final")
    s.add_argument("--batch-size", type=int, default=cw.DEFAULT_BATCH_SIZE)
    s.add_argument("--precision", choices=cw.PRECISIONS, default="fp32")
    s.add_argument("--device", choices=cw.DEVICES, default="cuda")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (CometStageError, ev.EvalStepError, cw.WorkerError) as exc:
        print(f"comet_stage {args.cmd}: FAILED: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "install":
        install_comet(args.venv, args.constraints, args.mode)
    elif args.cmd == "pull":
        runs = {r: PINNED_RUNS[r] for r in args.run} if args.run else None
        pull_runs(args.eval_root, args.hf_repo, runs=runs)
    elif args.cmd == "score":
        s = score_stage(
            args.eval_root,
            venv=args.venv,
            batch_size=args.batch_size,
            precision=args.precision,
            device=args.device,
            model_dir=args.model_dir,
            chunk_size=args.chunk_size,
        )
        print(
            f"comet: scored {s['n_sets']} sets, {s['n_segments']} segments, "
            f"{s['n_distinct_triples_all_sets']} distinct triples -> "
            f"{comet_root(args.eval_root) / SUMMARY_NAME}"
        )
    elif args.cmd == "upload":
        upload_comet(ev._hf_api(), args.repo, args.eval_root)
    elif args.cmd == "plan":
        lines = describe_stage(
            args.eval_root,
            args.runs_root,
            batch_size=args.batch_size,
            precision=args.precision,
            device=args.device,
        )
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
