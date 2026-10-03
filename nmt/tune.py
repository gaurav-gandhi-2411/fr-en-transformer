from __future__ import annotations

# Decoding-tuning CLI: alpha in {0.6, 0.8, 1.0, 1.2} x beam in {1, 4, 5} is
# searched first, then the segmentation threshold T is tuned separately (holding the winning
# alpha/beam fixed) over the T grid {64, 128, 192, 256}, compared against "off" (no segmentation
# at all, always decoded and scored as the candidate named `no_segmentation`). Every candidate
# is decoded with `nmt.translate.Translator` and scored exclusively through
# `nmt.selection.select`/`selection_objective` -- this module loads its eval sentences ONLY via
# `nmt.selection.load_selection_set`/`limited_selection_set`, never a path or loader of its own,
# so the same structural restriction `nmt/selection.py` enforces applies here too (checked by
# `tests/test_selection.py`'s source scan, extended to cover this file).
#
# CLI: `python -m nmt.tune --model DIR --out reports/<run>/selection_grid.json
#   [--alphas 0.6 0.8 1.0 1.2] [--beams 1 4 5] [--seg-thresholds 64 128 192 256]
#   [--limit-e1 N --limit-e2 N] [--batch-size 16]`
import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmt.selection import (
    SelectionSet,
    limited_selection_set,
    load_selection_set,
    selection_objective,
)
from nmt.translate import Translator

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALPHAS: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
DEFAULT_BEAMS: tuple[int, ...] = (1, 4, 5)
# T in {64, 128, 192, 256} source subword tokens; "off" is not a number here: it is the
# always-present `no_segmentation` candidate of `_segmentation_tune` (NO_SEGMENTATION_KEY), so
# the full grid is T in {64, 128, 192, 256, off}.
DEFAULT_SEG_THRESHOLDS: tuple[int, ...] = (64, 128, 192, 256)
NO_SEGMENTATION_KEY = "no_segmentation"


def _git_sha() -> str | None:
    """Best-effort git SHA for run provenance; tolerates failure (mirrors `nmt.train._git_sha`)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _sha256_of_model(model_dir: Path) -> str | None:
    """sha256 of the exported model's weight file, for provenance. `None` if the directory has no
    `model.safetensors` (e.g. a Hub repo id was passed instead of a local export directory)."""
    weights_path = Path(model_dir) / "model.safetensors"
    if not weights_path.is_file():
        return None
    h = hashlib.sha256()
    with weights_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_ids(
    translator: Translator,
    ids: Sequence[str],
    sources: Sequence[str],
    beam: int,
    alpha: float,
    batch_size: int,
    segment_threshold: int | None,
) -> tuple[dict[str, str], float]:
    """Translate `sources` (paired with `ids`) and time the call. Returns `({id: hyp}, seconds)`."""
    t0 = time.monotonic()
    translations = translator.translate(
        list(sources),
        batch_size=batch_size,
        beam=beam,
        alpha=alpha,
        segment_threshold=segment_threshold,
    )
    elapsed = time.monotonic() - t0
    return dict(zip(ids, translations, strict=True)), elapsed


@dataclass
class _AlphaBeamResult:
    """Everything the alpha x beam grid search produced. `decoded` is kept out of the written
    JSON report (it holds full per-candidate prediction dicts, not a summary) but is reused by the
    segmentation-tuning stage so the winning alpha/beam's predictions never need re-decoding."""

    grid: list[dict[str, float | int]]
    scores: dict[str, dict[str, float]]
    timings_seconds: dict[str, float]
    deduped_as: dict[str, str]
    best_key: str
    best_alpha: float
    best_beam: int
    decoded: dict[str, dict[str, str]]


def _alpha_beam_grid_search(
    translator: Translator,
    e1: SelectionSet,
    e2: SelectionSet,
    alphas: Sequence[float],
    beams: Sequence[int],
    batch_size: int,
) -> _AlphaBeamResult:
    """Search every nominal `(alpha, beam)` combo, scored via `selection_objective` on the
    combined E1+E2 sentences. Greedy (`beam=1`) ignores alpha entirely (a single-candidate beam
    has no length-penalty choice to make), so it is decoded exactly ONCE per grid search -- not
    once per nominal alpha value -- and every `alpha=*_beam=1` grid point reuses that one decode
    (`deduped_as` records which decode key each grid point actually came from)."""
    ids = list(e1.ids) + list(e2.ids)
    sources = list(e1.sources) + list(e2.sources)

    grid: list[dict[str, float | int]] = []
    decoded: dict[str, dict[str, str]] = {}
    timings: dict[str, float] = {}
    scores: dict[str, dict[str, float]] = {}
    deduped_as: dict[str, str] = {}
    key_to_params: dict[str, tuple[float, int]] = {}

    for beam in beams:
        if beam == 1:
            decode_key = f"beam={beam}"
            preds, elapsed = _decode_ids(
                translator, ids, sources, beam, alphas[0], batch_size, segment_threshold=None
            )
            decoded[decode_key] = preds
            timings[decode_key] = round(elapsed, 3)
            for alpha in alphas:
                grid_key = f"alpha={alpha}_beam={beam}"
                grid.append({"alpha": alpha, "beam": beam})
                scores[grid_key] = selection_objective(preds, e1, e2)
                deduped_as[grid_key] = decode_key
                key_to_params[grid_key] = (alpha, beam)
        else:
            for alpha in alphas:
                grid_key = f"alpha={alpha}_beam={beam}"
                grid.append({"alpha": alpha, "beam": beam})
                preds, elapsed = _decode_ids(
                    translator, ids, sources, beam, alpha, batch_size, segment_threshold=None
                )
                decoded[grid_key] = preds
                timings[grid_key] = round(elapsed, 3)
                scores[grid_key] = selection_objective(preds, e1, e2)
                key_to_params[grid_key] = (alpha, beam)

    best_key = max(scores, key=lambda k: scores[k]["objective"])
    best_alpha, best_beam = key_to_params[best_key]
    return _AlphaBeamResult(
        grid=grid,
        scores=scores,
        timings_seconds=timings,
        deduped_as=deduped_as,
        best_key=best_key,
        best_alpha=best_alpha,
        best_beam=best_beam,
        decoded=decoded,
    )


def _segmentation_tune(
    translator: Translator,
    e1: SelectionSet,
    e2: SelectionSet,
    fixed_e1_preds: dict[str, str],
    best_alpha: float,
    best_beam: int,
    seg_thresholds: Sequence[int],
    batch_size: int,
) -> dict[str, Any]:
    """Tune the segmentation threshold T on E2 only, after alpha/beam are already chosen: E1's
    predictions are held fixed (decoded once, at the winning alpha/beam, no segmentation), and
    only E2 is redecoded per candidate T -- "off" (no segmentation at all, key
    `NO_SEGMENTATION_KEY`) is ALWAYS included as one of the compared candidates, whatever
    `seg_thresholds` holds, and is not assumed better or worse. Still scored via
    `selection_objective` (which needs both E1 and E2), so a T change's effect on the objective is
    attributable to E2 alone."""
    scores: dict[str, dict[str, float]] = {}
    timings: dict[str, float] = {}
    key_to_threshold: dict[str, int | None] = {}

    def _score(label: str, e2_preds: dict[str, str], threshold: int | None, elapsed: float) -> None:
        combined = dict(fixed_e1_preds)
        combined.update(e2_preds)
        scores[label] = selection_objective(combined, e1, e2)
        timings[label] = round(elapsed, 3)
        key_to_threshold[label] = threshold

    e2_preds_none, elapsed_none = _decode_ids(
        translator, e2.ids, e2.sources, best_beam, best_alpha, batch_size, segment_threshold=None
    )
    _score(NO_SEGMENTATION_KEY, e2_preds_none, None, elapsed_none)

    for threshold in seg_thresholds:
        e2_preds_t, elapsed_t = _decode_ids(
            translator,
            e2.ids,
            e2.sources,
            best_beam,
            best_alpha,
            batch_size,
            segment_threshold=threshold,
        )
        _score(f"T={threshold}", e2_preds_t, threshold, elapsed_t)

    best_key = max(scores, key=lambda k: scores[k]["objective"])
    return {
        "seg_thresholds": list(seg_thresholds),
        "scores": scores,
        "timings_seconds": timings,
        "best": best_key,
        "best_segment_threshold": key_to_threshold[best_key],
    }


def run_tune(
    model_dir: Path,
    out_path: Path,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    beams: Sequence[int] = DEFAULT_BEAMS,
    seg_thresholds: Sequence[int] = DEFAULT_SEG_THRESHOLDS,
    limit_e1: int | None = None,
    limit_e2: int | None = None,
    batch_size: int = 16,
    translator: Translator | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full decoding-tuning path and write it to `out_path`: the alpha x
    beam grid (full score table, winner, per-decode timings, the greedy dedup note), then the
    segmentation-threshold search on top of the winning alpha/beam, then the combined winner,
    model sha256 and git SHA. `limit_e1`/`limit_e2` decode/score only the first N sentences of
    each set (a verified prefix, via `nmt.selection.limited_selection_set` -- never a different or
    tampered set) to keep CPU-only runs fast; `None` (default) uses the whole set. Always runs on
    `Translator.from_pretrained`'s own default compute placement (CUDA if present, else CPU).
    `translator` (default None = load `model_dir`) lets a caller tune an already-built translator,
    e.g. an ensemble (nmt.final_all); then `model_dir` is only a label, `model_sha256` is None and
    `extra` (extra top-level result keys, e.g. the member list and their hashes) is recorded."""
    provided = translator
    e1 = limited_selection_set(load_selection_set("e1"), limit_e1)
    e2 = limited_selection_set(load_selection_set("e2"), limit_e2)

    if translator is None:
        translator = Translator.from_pretrained(str(model_dir))

    ab = _alpha_beam_grid_search(translator, e1, e2, alphas, beams, batch_size)
    # A greedy winner's predictions live under its shared decode key ("beam=1"), not its grid key.
    best_decode_key = ab.deduped_as.get(ab.best_key, ab.best_key)
    fixed_e1_preds = {i: ab.decoded[best_decode_key][i] for i in e1.ids}
    seg = _segmentation_tune(
        translator, e1, e2, fixed_e1_preds, ab.best_alpha, ab.best_beam, seg_thresholds, batch_size
    )

    result: dict[str, Any] = {
        "model_dir": str(model_dir),
        "model_sha256": _sha256_of_model(Path(model_dir)) if provided is None else None,
        "git_sha": _git_sha(),
        "limit_e1": limit_e1,
        "limit_e2": limit_e2,
        "n_e1": len(e1),
        "n_e2": len(e2),
        "batch_size": batch_size,
        "alpha_beam": {
            "grid": ab.grid,
            "scores": ab.scores,
            "timings_seconds": ab.timings_seconds,
            "deduped_as": ab.deduped_as,
            "best": ab.best_key,
        },
        "segmentation": seg,
        "winner": {
            "alpha": ab.best_alpha,
            "beam": ab.best_beam,
            "segment_threshold": seg["best_segment_threshold"],
        },
        "fallback_counts": {
            "beam": translator.stats.n_beam,
            "greedy": translator.stats.n_greedy_fallback,
            "copy": translator.stats.n_copy_fallback,
        },
    }
    result.update(extra or {})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoding-tuning grid: alpha x beam, then segmentation threshold, on E1/E2 "
        "only (spec section 7)."
    )
    parser.add_argument("--model", required=True, type=Path, help="Exported model directory.")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the grid report.")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--beams", type=int, nargs="+", default=list(DEFAULT_BEAMS))
    parser.add_argument(
        "--seg-thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEG_THRESHOLDS),
        help="Segmentation thresholds T (source subword tokens) to compare; the 'off' candidate "
        f"({NO_SEGMENTATION_KEY}) is always added. Default: %(default)s.",
    )
    parser.add_argument("--limit-e1", type=int, default=None)
    parser.add_argument("--limit-e2", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_tune(
        args.model,
        args.out,
        alphas=tuple(args.alphas),
        beams=tuple(args.beams),
        seg_thresholds=tuple(args.seg_thresholds),
        limit_e1=args.limit_e1,
        limit_e2=args.limit_e2,
        batch_size=args.batch_size,
    )
    w = result["winner"]
    print(
        f"tune: wrote {args.out} -- winner alpha={w['alpha']} beam={w['beam']} "
        f"segment_threshold={w['segment_threshold']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
