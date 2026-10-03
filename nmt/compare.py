from __future__ import annotations

# Paired A/B comparison CLI: paired bootstrap resampling (Koehn 2004) reporting Delta, CI and p
# for ablations and decoding options. Reads the
# `<split>_predictions.json` files two `evaluate` runs already wrote, so a comparison never
# re-decodes and always scores exactly the predictions behind each run's eval.json. Delta is
# A - B per split (official BLEU and chrF), plus per slice (dev's official slices; E2-synth's
# char-length buckets) whenever a split has more than one.
#
# E2-synth (the synthetic long-input probe, nmt/data/e2synth.py) is compared too whenever BOTH
# dirs hold `e2synth_predictions.json`, unless `--no-e2synth`; it stays a separate, labelled
# split and is never pooled into another set.
#
# `--objective` adds a paired bootstrap of the SELECTION OBJECTIVE (nmt.selection's
# 0.4*BLEU(E1+E2) + 0.4*chrF(E1+E2) + 0.2*chrF(E1), official scorer implementations) between the
# two dirs, as `"selection_objective"` in the JSON. This module is an analysis tool, not part of
# the selection path: it reads finished predictions and never feeds a choice back to training or
# tuning.
#
# CLI: `python -m nmt.compare --a DIR_A --b DIR_B --out compare.json
#   [--splits dev e1 e2 e3] [--n-bootstrap 1000] [--seed 1234] [--objective] [--no-e2synth]`
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from nmt.evaluate import (
    SYNTHETIC_SPLIT_LABELS,
    _bleu_from_aggregated,
    _bleu_point_from_stats,
    _bleu_sentence_stats,
    _BleuStats,
    _chrf_point_from_values,
    _chrf_sentence_stats,
    _official_metric_fn,
    load_split,
    paired_bootstrap,
)

DEFAULT_SPLITS: tuple[str, ...] = ("dev", "e1", "e2", "e3")
METRICS: tuple[str, ...] = ("bleu", "chrf")

# Selection-objective weights (mirror nmt.selection.selection_objective).
_W_BLEU_UNION, _W_CHRF_UNION, _W_CHRF_E1 = 0.40, 0.40, 0.20


def _load_preds(eval_dir: Path, split: str) -> dict[str, str]:
    path = Path(eval_dir) / f"{split}_predictions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _has_preds(eval_dir: Path, split: str) -> bool:
    return (Path(eval_dir) / f"{split}_predictions.json").is_file()


def _paired(
    hyps_a: list[str], hyps_b: list[str], refs: list[str], n_bootstrap: int, seed: int
) -> dict[str, dict[str, Any]]:
    return {
        m: paired_bootstrap(hyps_a, hyps_b, refs, _official_metric_fn(m), n_bootstrap, seed)
        for m in METRICS
    }


# -------------------------------------------------------------------------------------------
# Selection-objective paired bootstrap
# -------------------------------------------------------------------------------------------


def _objective_inputs(eval_dir: Path, split: str) -> tuple[list[str], list[str]]:
    """`(hyps, refs)` of `split` in the split's inputs order (the order `nmt.selection` uses)."""
    inputs, labels = load_split(split)
    label_by_id = {r["id"]: r["reference"] for r in labels}
    pred = _load_preds(eval_dir, split)
    ids = [r["id"] for r in inputs]
    missing = [i for i in ids if i not in pred]
    if missing:
        raise ValueError(f"{split}: {len(missing)} ids missing from {eval_dir}, e.g. {missing[:3]}")
    return [pred[i] for i in ids], [label_by_id[i] for i in ids]


def _objective_point_from_stats(
    bleu1: _BleuStats, chrf1: np.ndarray, bleu2: _BleuStats, chrf2: np.ndarray
) -> dict[str, float]:
    """The selection objective from the already-cached per-sentence stats (no second tokenization
    pass). Union BLEU sums E1+E2's integer stats; union chrF is Python's left-to-right `sum` over
    the E1-then-E2 sentence values -- the same lists, order and operations as
    `official.score_slice(E1 + E2)` inside `nmt.selection.selection_objective`, so the value is
    bit-for-bit identical (checked in tests/test_compare.py)."""
    union = _BleuStats(
        hyp_len=np.concatenate([bleu1.hyp_len, bleu2.hyp_len]),
        ref_len=np.concatenate([bleu1.ref_len, bleu2.ref_len]),
        match=np.concatenate([bleu1.match, bleu2.match]),
        total=np.concatenate([bleu1.total, bleu2.total]),
    )
    bleu_union = _bleu_point_from_stats(union)
    chrf_union = _chrf_point_from_values(np.concatenate([chrf1, chrf2]))
    chrf_e1 = _chrf_point_from_values(chrf1)
    return {
        "objective": _W_BLEU_UNION * bleu_union + _W_CHRF_UNION * chrf_union + _W_CHRF_E1 * chrf_e1,
        "bleu_union": bleu_union,
        "chrf_union": chrf_union,
        "chrf_e1": chrf_e1,
    }


def _objective_point(
    hyps_e1: list[str], refs_e1: list[str], hyps_e2: list[str], refs_e2: list[str]
) -> dict[str, float]:
    """String-input form of `_objective_point_from_stats` (builds the stats itself)."""
    return _objective_point_from_stats(
        _bleu_sentence_stats(hyps_e1, refs_e1),
        _chrf_sentence_stats(hyps_e1, refs_e1),
        _bleu_sentence_stats(hyps_e2, refs_e2),
        _chrf_sentence_stats(hyps_e2, refs_e2),
    )


def _objective_samples(
    bleu1: _BleuStats,
    chrf1: np.ndarray,
    bleu2: _BleuStats,
    chrf2: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
) -> np.ndarray:
    """Selection objective on every resample, vectorized from per-sentence sufficient statistics.
    `idx1`/`idx2`: `(R, |E1|)` / `(R, |E2|)` indices drawn per set (stratified), so the union holds
    exactly |E1|+|E2| sentences in every resample and its official chrF (a sentence mean) is the
    pooled sum over that fixed count."""
    match = bleu1.match[idx1].sum(axis=1) + bleu2.match[idx2].sum(axis=1)
    total = bleu1.total[idx1].sum(axis=1) + bleu2.total[idx2].sum(axis=1)
    hyp_len = bleu1.hyp_len[idx1].sum(axis=1) + bleu2.hyp_len[idx2].sum(axis=1)
    ref_len = bleu1.ref_len[idx1].sum(axis=1) + bleu2.ref_len[idx2].sum(axis=1)
    bleu_union = _bleu_from_aggregated(match, total, hyp_len, ref_len)
    chrf1_sum = chrf1[idx1].sum(axis=1)
    chrf_union = (chrf1_sum + chrf2[idx2].sum(axis=1)) / (idx1.shape[1] + idx2.shape[1])
    chrf_e1 = chrf1_sum / idx1.shape[1]
    return _W_BLEU_UNION * bleu_union + _W_CHRF_UNION * chrf_union + _W_CHRF_E1 * chrf_e1


def paired_bootstrap_objective(
    hyps_a: tuple[list[str], list[str]],
    hyps_b: tuple[list[str], list[str]],
    refs: tuple[list[str], list[str]],
    n_bootstrap: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Paired bootstrap of the selection objective, A - B. `hyps_a`/`hyps_b`/`refs` are
    `(E1 list, E2 list)`. Each resample draws |E1| indices from E1 and |E2| indices from E2
    (stratified, with replacement) and uses the SAME indices for A and B. Reports the point delta
    (identical to `selection_objective(A) - selection_objective(B)`), a 95% percentile CI over the
    resampled deltas, and a one-sided p (fraction of resamples where B >= A, as in
    `evaluate.paired_bootstrap`)."""
    (h1a, h2a), (h1b, h2b), (r1, r2) = hyps_a, hyps_b, refs
    n1, n2 = len(r1), len(r2)
    if n1 == 0 or n2 == 0:
        raise ValueError("selection-objective bootstrap needs non-empty E1 and E2")
    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n1, size=(n_bootstrap, n1))
    idx2 = rng.integers(0, n2, size=(n_bootstrap, n2))
    samples = {}
    points = {}
    for key, (h1, h2) in (("a", (h1a, h2a)), ("b", (h1b, h2b))):
        # Stats computed once per (system, set); the point estimate and every resample reuse them.
        bleu1, chrf1 = _bleu_sentence_stats(h1, r1), _chrf_sentence_stats(h1, r1)
        bleu2, chrf2 = _bleu_sentence_stats(h2, r2), _chrf_sentence_stats(h2, r2)
        points[key] = _objective_point_from_stats(bleu1, chrf1, bleu2, chrf2)
        samples[key] = _objective_samples(bleu1, chrf1, bleu2, chrf2, idx1, idx2)
    point_a, point_b = points["a"], points["b"]
    deltas = samples["a"] - samples["b"]
    count_b_ge_a = int(np.sum(deltas <= 0))
    deltas = np.sort(deltas)
    return {
        "a": point_a,
        "b": point_b,
        "delta": point_a["objective"] - point_b["objective"],
        "ci_low": float(deltas[int(0.025 * n_bootstrap)]),
        "ci_high": float(deltas[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]),
        "p_value": count_b_ge_a / n_bootstrap,
        "n_resamples": n_bootstrap,
        "n_e1": n1,
        "n_e2": n2,
        "formula": "0.4*BLEU(E1+E2) + 0.4*chrF(E1+E2) + 0.2*chrF(E1), official score.py",
        "resampling": "stratified: |E1| drawn from E1 and |E2| from E2 per resample, same "
        "indices for A and B",
    }


# -------------------------------------------------------------------------------------------
# Per-split comparison
# -------------------------------------------------------------------------------------------


def compare_eval_dirs(
    dir_a: Path,
    dir_b: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    n_bootstrap: int = 1000,
    seed: int = 1234,
    objective: bool = False,
    include_e2synth: bool = True,
) -> dict[str, Any]:
    """Paired bootstrap of run A vs run B on every split in `splits`. Both runs must cover the
    split's full id set -- a missing id is an error, not a silently empty hypothesis. E2-synth is
    appended when `include_e2synth` and both dirs hold its predictions (its entry is flagged
    `synthetic`, with one paired test per char bucket). `objective=True` adds the paired bootstrap
    of the selection objective under `"selection_objective"` (needs E1 and E2 in both dirs)."""
    for synth in SYNTHETIC_SPLIT_LABELS:
        if (
            include_e2synth
            and synth not in splits
            and _has_preds(dir_a, synth)
            and _has_preds(dir_b, synth)
        ):
            splits = (*splits, synth)
    result: dict[str, Any] = {
        "a": str(dir_a),
        "b": str(dir_b),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "splits": {},
    }
    for split in splits:
        _, labels = load_split(split)
        pred_a, pred_b = _load_preds(dir_a, split), _load_preds(dir_b, split)
        ids = [r["id"] for r in labels]
        missing = [i for i in ids if i not in pred_a or i not in pred_b]
        if missing:
            raise ValueError(f"{split}: {len(missing)} ids missing from A or B, e.g. {missing[:3]}")
        refs = [r["reference"] for r in labels]
        entry: dict[str, Any] = {
            "n": len(ids),
            "overall": _paired(
                [pred_a[i] for i in ids], [pred_b[i] for i in ids], refs, n_bootstrap, seed
            ),
        }
        by_slice: dict[str, list[int]] = defaultdict(list)
        for k, row in enumerate(labels):
            by_slice[row["slice"]].append(k)
        if len(by_slice) > 1:
            entry["by_slice"] = {
                name: _paired(
                    [pred_a[ids[k]] for k in idx],
                    [pred_b[ids[k]] for k in idx],
                    [refs[k] for k in idx],
                    n_bootstrap,
                    seed,
                )
                for name, idx in by_slice.items()
            }
        if split in SYNTHETIC_SPLIT_LABELS:
            entry["synthetic"] = True
            entry["label"] = SYNTHETIC_SPLIT_LABELS[split]
        result["splits"][split] = entry
    if objective:
        e1_a, e1_refs = _objective_inputs(dir_a, "e1")
        e2_a, e2_refs = _objective_inputs(dir_a, "e2")
        e1_b, _ = _objective_inputs(dir_b, "e1")
        e2_b, _ = _objective_inputs(dir_b, "e2")
        result["selection_objective"] = paired_bootstrap_objective(
            (e1_a, e2_a), (e1_b, e2_b), (e1_refs, e2_refs), n_bootstrap, seed
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap A/B comparison (spec §8).")
    parser.add_argument("--a", required=True, type=Path, help="Eval dir of system A.")
    parser.add_argument("--b", required=True, type=Path, help="Eval dir of system B.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--objective",
        action="store_true",
        help="Also run the paired bootstrap of the selection objective (needs e1+e2 predictions).",
    )
    parser.add_argument(
        "--no-e2synth",
        action="store_true",
        help="Do not auto-include E2-synth when both dirs have e2synth_predictions.json.",
    )
    args = parser.parse_args(argv)
    result = compare_eval_dirs(
        args.a,
        args.b,
        tuple(args.splits),
        args.n_bootstrap,
        args.seed,
        objective=args.objective,
        include_e2synth=not args.no_e2synth,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, entry in result["splits"].items():
        c = entry["overall"]["chrf"]
        print(
            f"{split}: chrF delta={c['delta']:+.3f} "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.3f}"
        )
    if "selection_objective" in result:
        o = result["selection_objective"]
        print(
            f"selection objective: delta={o['delta']:+.3f} "
            f"[{o['ci_low']:+.3f}, {o['ci_high']:+.3f}] p={o['p_value']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
