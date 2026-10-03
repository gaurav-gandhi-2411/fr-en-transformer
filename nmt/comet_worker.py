from __future__ import annotations

# The COMET-22 scoring WORKER of the final_all Colab session (nmt/comet_stage.py drives it).
# It runs under the interpreter of the COMET venv (python -m venv --system-site-packages: Colab's
# torch stays visible, unbabel-comet and its older pins live only in the venv), loads
# Unbabel/wmt22-comet-da ONCE at a pinned revision, and scores a list of distinct
# {src, mt, ref} triples in fixed-size chunks. Every finished chunk is written atomically to
# <cache-dir>/chunk_NNNNN.json, so a disconnect loses at most one chunk and a re-run skips the
# finished ones (the model is not even loaded when every chunk is already there).
#
# The orchestrator feeds the whole canonical triple list every time, so chunk k always holds the
# same triples and the same batch composition: scores do not depend on where a previous run
# stopped. Top level imports only the stdlib: torch / comet are imported inside the functions
# that need them, so the pure parts are unit-testable without them.
#
# CLI: python -m nmt.comet_worker --in triples.json --cache-dir DIR --meta-out meta.json
#        [--chunk-size 4096] [--batch-size 64] [--device auto|cuda|cpu]
#        [--precision fp32|bf16|fp16] [--seed 1234]
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

COMET_MODEL = "Unbabel/wmt22-comet-da"
# Pinned: the hub head at the time of writing (last modified 2025-02-26); the model card says
# apache-2.0. Looked up read-only with `HfApi().model_info`, not recalled.
COMET_MODEL_REVISION = "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
# The checkpoint's encoder is `xlm-roberta-large`; only its config + tokenizer are read (the
# weights come from the COMET checkpoint), pinned the same way. MIT licence per its card.
XLMR_REPO = "FacebookAI/xlm-roberta-large"
XLMR_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
XLMR_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model")
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 1234
PRECISIONS = ("fp32", "bf16", "fp16")
DEVICES = ("auto", "cuda", "cpu")
VERSION_PACKAGES = (
    "unbabel-comet",
    "torch",
    "transformers",
    "pytorch-lightning",
    "torchmetrics",
    "numpy",
    "huggingface-hub",
    "protobuf",
)

# What only a call that actually loads the model can record; kept across no-op re-runs.
LOAD_FIELDS = (
    "model",
    "model_revision",
    "encoder",
    "encoder_revision",
    "model_class",
    "class_identifier",
    "device",
)

Triple = dict[str, str]


class WorkerError(RuntimeError):
    """A failure the orchestrator should print as is (missing CUDA, bad input, ...)."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_sha256(triples: Sequence[Triple]) -> str:
    """Digest of one chunk's input: a stored chunk is only reused when this still matches."""
    blob = json.dumps(
        [[t["src"], t["mt"], t["ref"]] for t in triples], ensure_ascii=False, separators=(",", ":")
    )
    return _sha256_bytes(blob.encode("utf-8"))


def chunk_bounds(n: int, chunk_size: int) -> list[tuple[int, int]]:
    """[start, end) bounds covering range(n) in order, each at most chunk_size long."""
    if chunk_size < 1:
        raise WorkerError(f"chunk size must be >= 1, got {chunk_size}")
    return [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]


def chunk_path(cache_dir: Path, k: int) -> Path:
    return Path(cache_dir) / f"chunk_{k:05d}.json"


def _write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written chunk


def read_chunk(path: Path, triples: Sequence[Triple]) -> list[float] | None:
    """The scores of a finished chunk, or None unless the file exists, parses, is for exactly
    these triples (digest) and holds one finite float per triple (fail closed)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("chunk_sha256") != chunk_sha256(triples):
        return None
    scores = data.get("scores")
    if not (isinstance(scores, list) and len(scores) == len(triples)):
        return None
    if not all(
        isinstance(s, (int, float))
        and not isinstance(s, bool)
        and s == s
        and abs(s) != float("inf")
        for s in scores
    ):
        return None
    return [float(s) for s in scores]


def run_chunks(
    triples: Sequence[Triple],
    cache_dir: Path,
    chunk_size: int,
    get_scorer: Callable[[], Callable[[list[Triple]], list[float]]],
    log: Callable[[str], None] = print,
    chunk_meta: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score every chunk that has no valid file yet. `get_scorer` is called lazily, at most once,
    so a complete cache never loads the model. `chunk_meta` (called after the scorer is
    ready) is stored in every chunk file it scores, as provenance (model record, device,
    precision, batch size, library versions): a killed run whose worker_meta.json was
    never written can still be re-derived from the chunks. Batch size is recorded but NOT
    part of a chunk's validity: it only changes padding, i.e. scores at the ~1e-7 level
    (up to 6e-7 measured between chunk/batch compositions), never the model or the precision.
    Returns {"chunks", "skipped", "scored_triples",
    "wall_seconds", "chunk_wall_seconds": {k: seconds}} for the chunks scored by THIS call."""
    bounds = chunk_bounds(len(triples), chunk_size)
    scorer: Callable[[list[Triple]], list[float]] | None = None
    skipped, scored, wall = 0, 0, 0.0
    per_chunk: dict[str, float] = {}
    for k, (lo, hi) in enumerate(bounds):
        part = list(triples[lo:hi])
        path = chunk_path(cache_dir, k)
        if read_chunk(path, part) is not None:
            skipped += 1
            continue
        if scorer is None:
            scorer = get_scorer()
        t0 = time.monotonic()
        scores = scorer(part)
        seconds = time.monotonic() - t0
        if len(scores) != len(part):
            raise WorkerError(f"chunk {k}: {len(scores)} scores for {len(part)} triples")
        _write_json(
            path,
            {
                "chunk_sha256": chunk_sha256(part),
                "start": lo,
                "n": len(part),
                "scores": [float(s) for s in scores],
                "wall_seconds": seconds,
                "meta": chunk_meta() if chunk_meta else {},
            },
        )
        scored += len(part)
        wall += seconds
        per_chunk[str(k)] = seconds
        log(
            f"comet_worker: chunk {k + 1}/{len(bounds)} done ({hi}/{len(triples)} triples, "
            f"{len(part) / max(seconds, 1e-9):.1f} triples/s)"
        )
    return {
        "chunks": len(bounds),
        "skipped": skipped,
        "scored_triples": scored,
        "wall_seconds": wall,
        "chunk_wall_seconds": per_chunk,
    }


def library_versions() -> dict[str, str | None]:
    """Installed versions (package metadata, no import) of the packages that decide COMET scores."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in VERSION_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    """symlink, else hardlink, else copy (Windows without symlink rights; the file is 2.3 GB)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (os.symlink, os.link, shutil.copyfile):
        try:
            attempt(src, dst)
            return
        except OSError:
            continue
    raise WorkerError(f"cannot link or copy {src} to {dst}")


def prepare_pinned_model_dir(work_dir: Path, model_dir: Path, encoder_dir: Path) -> Path:
    """<work_dir>/checkpoints/model.ckpt (linked) + hparams.yaml whose `pretrained_model` points
    at the pinned local xlm-roberta-large snapshot instead of the moving hub name. Returns the
    checkpoint path. Only that one line of hparams.yaml changes."""
    work_dir = Path(work_dir)
    text = (Path(model_dir) / "hparams.yaml").read_text(encoding="utf-8")
    lines, replaced = [], False
    for line in text.splitlines():
        if line.startswith("pretrained_model:"):
            line, replaced = f"pretrained_model: {Path(encoder_dir).as_posix()}", True
        lines.append(line)
    if not replaced:
        raise WorkerError(f"{model_dir}/hparams.yaml has no pretrained_model line to pin")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "hparams.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ckpt = work_dir / "checkpoints" / "model.ckpt"
    if not ckpt.exists():
        _link_or_copy(Path(model_dir) / "checkpoints" / "model.ckpt", ckpt)
    return ckpt


def cpu_fallback_notice(device: str, gpus: int) -> str | None:
    """LOUD text when `auto` ended on the CPU (days instead of minutes for the real
    workload), else None. The notebook asks for `cuda`, which refuses instead."""
    if device == "auto" and not gpus:
        return (
            "!!! COMET WORKER: --device auto found NO CUDA: scoring on the CPU, a few "
            "triples/s (1.6-7.5 measured on laptop CPUs, small samples) instead of the ASSUMED "
            "50-150 on a GPU. Use --device cuda to refuse. !!!"
        )
    return None


def resolve_device(device: str, cuda_available: bool) -> int:
    """gpus argument of COMET's predict: 1 for CUDA, 0 for CPU. 'cuda' without CUDA is an error
    (never a silent CPU run that would take days); 'auto' falls back to CPU."""
    if device not in DEVICES:
        raise WorkerError(f"device {device!r} is not one of {DEVICES}")
    if device == "cuda" and not cuda_available:
        raise WorkerError("--device cuda but torch.cuda.is_available() is False")
    return 1 if (device == "cuda" or (device == "auto" and cuda_available)) else 0


def load_model(work_dir: Path) -> tuple[Any, dict[str, Any]]:
    """The pinned COMET model (weights from COMET_MODEL_REVISION, encoder config + tokenizer from
    XLMR_REVISION), kept in fp32; plus a record of what was loaded. Reduced precision is applied
    per call with torch.autocast (make_scorer): casting the weights with .half() / .to(bfloat16)
    fails inside COMET (the layer-mix estimator mixes dtypes), measured on CPU."""
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(COMET_MODEL, revision=COMET_MODEL_REVISION))
    encoder_dir = Path(
        snapshot_download(XLMR_REPO, revision=XLMR_REVISION, allow_patterns=list(XLMR_FILES))
    )
    ckpt = prepare_pinned_model_dir(Path(work_dir) / "model", model_dir, encoder_dir)
    model = load_from_checkpoint(str(ckpt), reload_hparams=True, local_files_only=True)
    model.eval()
    record = {
        "model": COMET_MODEL,
        "model_revision": COMET_MODEL_REVISION,
        "encoder": XLMR_REPO,
        "encoder_revision": XLMR_REVISION,
        "model_class": type(model).__name__,
        "class_identifier": model.hparams.get("class_identifier"),
    }
    return model, record


def autocast_dtype(precision: str) -> str | None:
    """torch dtype name for autocast, None for fp32 (no autocast: the reference numerics)."""
    if precision not in PRECISIONS:
        raise WorkerError(f"precision {precision!r} is not one of {PRECISIONS}")
    return {"fp32": None, "bf16": "bfloat16", "fp16": "float16"}[precision]


def make_scorer(
    model: Any, batch_size: int, gpus: int, precision: str = "fp32"
) -> Callable[[list[Triple]], list[float]]:
    """triples -> COMET-22 scores (one per triple, in order). fp32 runs without autocast; bf16 /
    fp16 wrap predict in torch.autocast on the scoring device (weights stay fp32)."""
    import contextlib

    dtype_name = autocast_dtype(precision)

    def score(triples: list[Triple]) -> list[float]:
        import torch

        data = [{"src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triples]
        ctx: Any = contextlib.nullcontext()
        if dtype_name is not None:
            ctx = torch.autocast(
                device_type="cuda" if gpus else "cpu", dtype=getattr(torch, dtype_name)
            )
        with ctx:
            out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
        return [float(s) for s in out.scores]

    return score


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score {src, mt, ref} triples with COMET-22.")
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "comet_model",
        help="LOCAL disk dir for the 2.3 GB checkpoint link/copy (never Drive)",
    )
    p.add_argument("--meta-out", required=True, type=Path)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", choices=DEVICES, default="auto")
    p.add_argument("--precision", choices=PRECISIONS, default="fp32")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    triples = json.loads(args.in_path.read_text(encoding="utf-8"))
    if not (isinstance(triples, list) and all(isinstance(t, dict) for t in triples)):
        raise WorkerError(f"{args.in_path}: expected a JSON list of {{src, mt, ref}} objects")
    state: dict[str, Any] = {}

    def provenance() -> dict[str, Any]:
        return {
            "schema": 1,
            "chunk_size": args.chunk_size,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "seed": args.seed,
            "libraries": library_versions(),
        }

    def get_scorer() -> Callable[[list[Triple]], list[float]]:
        import torch

        gpus = resolve_device(args.device, torch.cuda.is_available())
        notice = cpu_fallback_notice(args.device, gpus)
        if notice:
            print(notice, file=sys.stderr, flush=True)
        if args.precision == "fp16" and not gpus:
            raise WorkerError("fp16 needs a GPU (CPU half-precision ops are not supported)")
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model, record = load_model(args.model_dir)
        state.update(record)
        state["device"] = f"cuda:0 ({torch.cuda.get_device_name(0)})" if gpus else "cpu"
        # written as soon as the model is loaded, before the first chunk (atomic)
        _write_json(args.meta_out, {**provenance(), **state, "model_loaded_this_call": True})
        return make_scorer(model, args.batch_size, gpus, args.precision)

    t0 = time.monotonic()
    stats = run_chunks(
        triples,
        args.cache_dir,
        args.chunk_size,
        get_scorer,
        chunk_meta=lambda: {**provenance(), **state},
    )
    try:  # a re-run that finds every chunk done loads no model: keep what the loading run saw
        previous = json.loads(args.meta_out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    kept = {k: previous[k] for k in LOAD_FIELDS if k in previous}
    meta = {
        **kept,
        "schema": 1,
        "n_triples": len(triples),
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "seed": args.seed,
        "libraries": library_versions(),
        "model_loaded_this_call": bool(state),
        "this_call_seconds": time.monotonic() - t0,
        **state,
        **stats,
    }
    _write_json(args.meta_out, meta)
    print(
        f"comet_worker: {stats['scored_triples']} triples scored in {stats['wall_seconds']:.1f} s "
        f"({stats['skipped']} of {stats['chunks']} chunks already done)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkerError as exc:
        print(f"comet_worker: FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
