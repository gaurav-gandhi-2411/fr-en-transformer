from __future__ import annotations

# Evaluation: runs the vendored official/score.py exactly as shipped (subprocess CLI, primary
# metric), re-derives the same numbers in-process via importlib (for per-sentence chrF and
# bootstrap resampling, without ever modifying official/score.py), sacreBLEU BLEU/chrF/chrF++
# with signatures, optional COMET-22 (eval-only, isolated env, never used for selection),
# bootstrap 95% CIs and paired bootstrap A/B comparisons, reported by slice / length bucket /
# E-set. Spec §8.
import importlib.util
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import sacrebleu
import sentencepiece as spm
import torch

from nmt.decode import greedy_decode
from nmt.translate import Translator, _detok, _pad_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCORE_PY = REPO_ROOT / "official" / "score.py"

LENGTH_BUCKET_LABELS = ("<=10", "11-20", "21-40", "41-80", ">80")

_official_module_cache: ModuleType | None = None


def load_official_module() -> ModuleType:
    """Import `official/score.py`'s functions in-process via `importlib`, without ever editing
    the vendored file (spec §2/§8: run it "exactly as shipped"; sha256-checked separately in
    `tests/test_official_scorer.py`). Cached after first import.
    """
    global _official_module_cache
    if _official_module_cache is None:
        spec = importlib.util.spec_from_file_location("official_score", OFFICIAL_SCORE_PY)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _official_module_cache = module
    return _official_module_cache


def run_official_scorer_cli(gold_path: Path, pred_path: Path, out_path: Path) -> dict[str, Any]:
    """Run `official/score.py` exactly as shipped via `subprocess` (the primary metric, spec §8),
    parsing the `--out` JSON report it writes."""
    subprocess.run(
        [
            sys.executable,
            str(OFFICIAL_SCORE_PY),
            "--gold",
            str(gold_path),
            "--pred",
            str(pred_path),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def compute_official_metrics(pred: dict[str, str], gold_rows: list[dict]) -> dict[str, Any]:
    """In-process re-derivation of `official/score.py::main`'s report (same `{"all", "OVERALL",
    "by_slice"}` shape), calling the *same* imported functions on the *same* inputs the CLI would
    see -- `tests/test_evaluate.py` checks this equals `run_official_scorer_cli`'s output exactly
    on a fixed prediction set (spec §12).
    """
    module = load_official_module()
    gold = {r["id"]: r for r in gold_rows}
    ids = list(gold)
    hyps = [pred.get(i, "") for i in ids]
    refs = [gold[i].get("reference", "") for i in ids]
    by: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for i in ids:
        h, r = pred.get(i, ""), gold[i].get("reference", "")
        sl = gold[i].get("slice", "unspecified")
        by[sl][0].append(h)
        by[sl][1].append(r)
    all_scores = module.score_slice(hyps, refs)
    by_slice = {k: module.score_slice(h, r) for k, (h, r) in by.items()}
    unseen = by_slice.get("unseen_domain")
    unseen_chrf = unseen["chrf"] if unseen else all_scores["chrf"]
    overall = 0.40 * all_scores["bleu"] + 0.40 * all_scores["chrf"] + 0.20 * unseen_chrf
    return {"all": all_scores, "OVERALL": overall, "by_slice": by_slice}


# -------------------------------------------------------------------------------------------
# sacreBLEU
# -------------------------------------------------------------------------------------------


def sacrebleu_metrics(hyps: list[str], refs: list[str]) -> dict[str, Any]:
    """sacreBLEU BLEU, chrF and chrF++ with their reproducibility signatures (spec §8)."""
    bleu = sacrebleu.BLEU()
    chrf = sacrebleu.CHRF()
    chrfpp = sacrebleu.CHRF(word_order=2)
    return {
        "bleu": {
            "score": bleu.corpus_score(hyps, [refs]).score,
            "signature": bleu.get_signature().format(),
        },
        "chrf": {
            "score": chrf.corpus_score(hyps, [refs]).score,
            "signature": chrf.get_signature().format(),
        },
        "chrf++": {
            "score": chrfpp.corpus_score(hyps, [refs]).score,
            "signature": chrfpp.get_signature().format(),
        },
    }


# -------------------------------------------------------------------------------------------
# Bootstrap statistics
# -------------------------------------------------------------------------------------------
# Bootstrap CIs resample *sentence indices* with replacement and recompute a corpus-level metric
# on each resample. Recomputing BLEU/chrF from raw strings (retokenizing, rebuilding n-gram
# Counters) on every resample is what made 1000 resamples on E1 (1940 sentences) take several
# minutes per split per metric (see PLAN.md's smoke-gate deviation note -- `n_bootstrap` had to be
# cut to 200/100 there). The fix: compute per-sentence *sufficient statistics* ONCE, using
# `official/score.py`'s own `wtok`/`ngrams`/`chrf_sentence` (via `load_official_module`, never
# reimplemented), then each resample is a vectorized numpy gather-and-sum/mean over those
# precomputed statistics instead of a re-tokenization.


@dataclass(frozen=True)
class _BleuStats:
    """Per-sentence BLEU sufficient statistics, accumulated exactly as `official/score.py`'s
    `corpus_bleu` accumulates them (same `wtok`/`ngrams` calls) but kept *per sentence* instead of
    summed across the whole corpus, so a bootstrap resample only needs to sum the sampled rows."""

    hyp_len: np.ndarray  # (n,) int64
    ref_len: np.ndarray  # (n,) int64
    match: np.ndarray  # (n, 4) int64 -- matched n-gram counts, n=1..4
    total: np.ndarray  # (n, 4) int64 -- total hyp n-gram counts, n=1..4


def _bleu_sentence_stats(hyps: list[str], refs: list[str], max_n: int = 4) -> _BleuStats:
    """Per-sentence `(hyp_len, ref_len, match[1..max_n], total[1..max_n])`, computed with the
    official module's own `wtok`/`ngrams` so tokenization is byte-for-byte identical to
    `corpus_bleu` -- this is the per-sentence breakdown of exactly what `corpus_bleu`'s loop
    accumulates into its corpus-level `match`/`total` arrays."""
    module = load_official_module()
    n = len(hyps)
    hyp_len = np.empty(n, dtype=np.int64)
    ref_len = np.empty(n, dtype=np.int64)
    match = np.empty((n, max_n), dtype=np.int64)
    total = np.empty((n, max_n), dtype=np.int64)
    for i, (h, r) in enumerate(zip(hyps, refs, strict=True)):
        ht, rt = module.wtok(h), module.wtok(r)
        hyp_len[i] = len(ht)
        ref_len[i] = len(rt)
        for k in range(max_n):
            n_gram = k + 1
            hn, rn = module.ngrams(ht, n_gram), module.ngrams(rt, n_gram)
            match[i, k] = sum(min(c, rn[g]) for g, c in hn.items())
            total[i, k] = max(len(ht) - n_gram + 1, 0)
    return _BleuStats(hyp_len=hyp_len, ref_len=ref_len, match=match, total=total)


def _bleu_from_aggregated(
    match: np.ndarray, total: np.ndarray, hyp_len: np.ndarray, ref_len: np.ndarray
) -> np.ndarray:
    """The exact `corpus_bleu` formula (official/score.py), vectorized over an arbitrary leading
    batch shape: `match`/`total` are `(..., 4)`, `hyp_len`/`ref_len` are `(...,)`. Reproduces every
    edge case verbatim: n==1's `1e-9` zero-match floor, the `+1` smoothing for n>=2, `total==0 ->
    0.0`, `min(precs) <= 0 -> 0`, and the brevity penalty's strict `hyp_len > ref_len` / `hyp_len
    == 0 -> 0.0` cases.
    """
    hyp_len = hyp_len.astype(np.float64)
    ref_len = ref_len.astype(np.float64)
    match = match.astype(np.float64)
    total = total.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        t0, m0 = total[..., 0], match[..., 0]
        p0 = np.where(t0 == 0, 0.0, np.where(m0 > 0, m0 / np.where(t0 == 0, 1.0, t0), 1e-9))
        tn, mn = total[..., 1:], match[..., 1:]
        pn = np.where(tn == 0, 0.0, (mn + 1.0) / np.where(tn == 0, 1.0, tn + 1.0))
        precs = np.concatenate([p0[..., None], pn], axis=-1)

        min_precs = precs.min(axis=-1)
        valid = (hyp_len > 0) & (min_precs > 0)
        safe_precs = np.where(valid[..., None], precs, 1.0)  # avoid log(0); masked out below
        geo = np.exp(np.mean(np.log(safe_precs), axis=-1))
        safe_hyp_len = np.where(hyp_len > 0, hyp_len, 1.0)  # avoid /0; masked out below
        bp = np.where(hyp_len > ref_len, 1.0, np.exp(1.0 - ref_len / safe_hyp_len))

    return np.where(valid, 100.0 * bp * geo, 0.0)


def _bleu_resample(stats: _BleuStats, idx: np.ndarray) -> np.ndarray:
    """`idx`: `(n_resamples, n)` sentence indices drawn with replacement. Gathers and sums each
    resample's per-sentence stats, then applies `_bleu_from_aggregated` -- the vectorized
    equivalent of calling `corpus_bleu` on each resampled string list."""
    return _bleu_from_aggregated(
        stats.match[idx].sum(axis=1),
        stats.total[idx].sum(axis=1),
        stats.hyp_len[idx].sum(axis=1),
        stats.ref_len[idx].sum(axis=1),
    )


def _chrf_sentence_stats(hyps: list[str], refs: list[str]) -> np.ndarray:
    """Per-sentence `chrf_sentence` values (official chrF is a plain sentence average, so this
    *is* the sufficient statistic -- no further reduction is needed to resample it)."""
    module = load_official_module()
    if not refs:
        return np.zeros(0, dtype=np.float64)
    return np.array(
        [module.chrf_sentence(h, r) for h, r in zip(hyps, refs, strict=True)], dtype=np.float64
    )


def _chrf_resample(chrf_vals: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`idx`: `(n_resamples, n)`. Official chrF is a sentence average, so a resample's corpus chrF
    is just the mean of the resampled per-sentence values."""
    return chrf_vals[idx].mean(axis=1)


@dataclass(frozen=True)
class _VectorizableMetric:
    """A metric function paired with the sufficient-statistics machinery needed to vectorize its
    own bootstrap resampling. `bootstrap_ci`/`paired_bootstrap` use the fast numpy path below
    whenever `metric_fn` is one of these (the official BLEU/chrF metrics, spec §8); any other
    callable (e.g. a caller's own ad hoc scorer, as used directly in some tests) falls back to the
    generic, slower, always-correct per-resample Python loop -- the public bootstrap API is
    unchanged either way.
    """

    name: str
    point_fn: Callable[[list[str], list[str]], float]
    stats_fn: Callable[[list[str], list[str]], Any]
    resample_fn: Callable[[Any, np.ndarray], np.ndarray]

    def __call__(self, hyps: list[str], refs: list[str]) -> float:
        return self.point_fn(hyps, refs)


def _chrf_point(hyps: list[str], refs: list[str]) -> float:
    if not refs:
        return 0.0
    return float(_chrf_sentence_stats(hyps, refs).mean())


_BLEU_METRIC = _VectorizableMetric(
    name="bleu",
    point_fn=lambda h, r: load_official_module().corpus_bleu(h, r),
    stats_fn=_bleu_sentence_stats,
    resample_fn=_bleu_resample,
)
_CHRF_METRIC = _VectorizableMetric(
    name="chrf", point_fn=_chrf_point, stats_fn=_chrf_sentence_stats, resample_fn=_chrf_resample
)


def _official_metric_fn(metric: str) -> _VectorizableMetric:
    if metric == "bleu":
        return _BLEU_METRIC
    if metric == "chrf":
        return _CHRF_METRIC
    raise ValueError(f"unknown metric: {metric!r}")


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """`(n_resamples, n)` sentence indices drawn with replacement (spec §8)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_ci(
    hyps: list[str],
    refs: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Bootstrap 95% CI (spec §8: 1000 resamples, seeded) for a corpus-level metric. Resamples
    *sentence indices* (paired hyp/ref) with replacement and recomputes the corpus metric on each
    resample -- correct for non-additive corpus metrics like BLEU, unlike averaging per-sentence
    values. When `metric_fn` is a `_VectorizableMetric` (the official BLEU/chrF), resampling is
    fully vectorized via precomputed sufficient statistics (1000 resamples on ~2000 sentences
    runs in well under a second); any other callable falls back to a plain Python loop.
    """
    n = len(refs)
    point = metric_fn(hyps, refs)
    if n == 0:
        return {
            "point": point,
            "ci_low": point,
            "ci_high": point,
            "n_resamples": n_resamples,
            "n": 0,
        }
    if isinstance(metric_fn, _VectorizableMetric):
        stats = metric_fn.stats_fn(hyps, refs)
        idx = _resample_indices(n, n_resamples, seed)
        samples = np.sort(metric_fn.resample_fn(stats, idx))
        lo = float(samples[int(0.025 * n_resamples)])
        hi = float(samples[min(n_resamples - 1, int(0.975 * n_resamples))])
    else:
        rng = random.Random(seed)
        samples_list = []
        for _ in range(n_resamples):
            resampled = [rng.randrange(n) for _ in range(n)]
            samples_list.append(
                metric_fn([hyps[i] for i in resampled], [refs[i] for i in resampled])
            )
        samples_list.sort()
        lo = samples_list[int(0.025 * n_resamples)]
        hi = samples_list[min(n_resamples - 1, int(0.975 * n_resamples))]
    # Degenerate groups (n==1, or a group whose resamples are all numerically identical) can have
    # `lo`/`hi` differ from `point` by float ULP noise -- e.g. the vectorized aggregate formula
    # and `metric_fn`'s own direct computation take different operation orders, so a single-index
    # resample is mathematically but not bit-for-bit equal to the point estimate. A percentile CI
    # must always bracket its own point estimate; clip to guarantee that invariant exactly.
    lo = min(lo, point)
    hi = max(hi, point)
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_resamples": n_resamples, "n": n}


def bootstrap_ci_official(
    hyps: list[str], refs: list[str], metric: str, n_resamples: int = 1000, seed: int = 1234
) -> dict[str, Any]:
    """`bootstrap_ci` specialized to the official scorer's own BLEU/chrF implementations."""
    return bootstrap_ci(hyps, refs, _official_metric_fn(metric), n_resamples, seed)


def bootstrap_ci_by_group(
    hyps: list[str],
    refs: list[str],
    groups: Sequence[str],
    metric: str,
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict[str, dict[str, Any]]:
    """Per-group bootstrap 95% CI for an official BLEU/chrF metric (spec §8: "per slice and
    metric"). `groups[i]` labels `hyps[i]`/`refs[i]`'s official slice, E-set or length bucket;
    each group's CI resamples only that group's own sentence indices (not the pooled corpus), so
    it reflects that slice's own sample size -- not a decomposition of one pooled CI. The point
    estimate per group is `official/score.py`'s own metric on that group (identical to
    `compute_official_metrics`'s `by_slice` values, just with a CI attached).
    """
    by: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for h, r, g in zip(hyps, refs, groups, strict=True):
        by[g][0].append(h)
        by[g][1].append(r)
    return {
        g: bootstrap_ci_official(hs, rs, metric, n_resamples, seed) for g, (hs, rs) in by.items()
    }


def bootstrap_official_overall(
    pred: dict[str, str], gold_rows: list[dict], n_resamples: int = 1000, seed: int = 1234
) -> dict[str, Any]:
    """Bootstrap 95% CI for the official OVERALL formula (0.4*BLEU + 0.4*chrF + 0.2*chrF(unseen))
    -- each resample draws sentence indices jointly (paired across slices) and recomputes OVERALL
    on that resample, so the CI reflects the actual composite metric, not three independent CIs.
    Vectorized via precomputed BLEU/chrF sufficient statistics (same mechanism as `bootstrap_ci`).
    """
    ids = [r["id"] for r in gold_rows]
    n = len(ids)
    hyps_all = [pred.get(i, "") for i in ids]
    refs_all = [r.get("reference", "") for r in gold_rows]
    is_unseen = np.array([r.get("slice", "unspecified") == "unseen_domain" for r in gold_rows])

    if n == 0:
        # The original per-resample implementation crashed here (`module.score_slice([], [])`
        # returns `None`, then `None["bleu"]` raises `TypeError`) -- never exercised in practice
        # (dev is never empty), but there is no reason to keep that crash; return a flat 0.0 CI.
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_resamples": n_resamples, "n": 0}

    bleu_stats = _bleu_sentence_stats(hyps_all, refs_all)
    chrf_vals = _chrf_sentence_stats(hyps_all, refs_all)

    bleu_point = float(
        _bleu_from_aggregated(
            bleu_stats.match.sum(axis=0),
            bleu_stats.total.sum(axis=0),
            bleu_stats.hyp_len.sum(),
            bleu_stats.ref_len.sum(),
        )
    )
    chrf_all_point = float(chrf_vals.mean())
    chrf_unseen_point = float(chrf_vals[is_unseen].mean()) if is_unseen.any() else chrf_all_point
    point = 0.40 * bleu_point + 0.40 * chrf_all_point + 0.20 * chrf_unseen_point

    idx = _resample_indices(n, n_resamples, seed)
    bleu_samples = _bleu_resample(bleu_stats, idx)
    chrf_gathered = chrf_vals[idx]  # (n_resamples, n)
    chrf_all_samples = chrf_gathered.mean(axis=1)
    unseen_mask = is_unseen[idx]  # (n_resamples, n) -- same resample indices, per spec §8
    unseen_count = unseen_mask.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chrf_unseen_masked = (chrf_gathered * unseen_mask).sum(axis=1) / np.where(
            unseen_count == 0, 1, unseen_count
        )
    chrf_unseen_samples = np.where(unseen_count > 0, chrf_unseen_masked, chrf_all_samples)
    overall_samples = np.sort(
        0.40 * bleu_samples + 0.40 * chrf_all_samples + 0.20 * chrf_unseen_samples
    )
    lo = float(overall_samples[int(0.025 * n_resamples)])
    hi = float(overall_samples[min(n_resamples - 1, int(0.975 * n_resamples))])
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_resamples": n_resamples, "n": n}


def paired_bootstrap(
    hyps_a: list[str],
    hyps_b: list[str],
    refs: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Paired bootstrap resampling (Koehn 2004) for an A/B comparison: Delta = metric(A) -
    metric(B), a 95% CI on Delta, and a one-sided p-value (the fraction of resamples where B's
    score meets or exceeds A's -- evidence *against* "A is better"). Used for ablations and
    decoding-option comparisons (spec §8/§9). The SAME resample indices are used for both systems
    in every resample (required for a valid paired test) -- true of both the vectorized and the
    generic fallback path below. Fast-pathed via sufficient statistics when `metric_fn` is a
    `_VectorizableMetric` (the official BLEU/chrF); any other callable falls back to a plain
    Python loop.
    """
    n = len(refs)
    point_a = metric_fn(hyps_a, refs)
    point_b = metric_fn(hyps_b, refs)
    delta_point = point_a - point_b
    if n == 0:
        return {
            "delta": delta_point,
            "ci_low": delta_point,
            "ci_high": delta_point,
            "p_value": 1.0,
            "n_resamples": n_resamples,
        }
    if isinstance(metric_fn, _VectorizableMetric):
        idx = _resample_indices(n, n_resamples, seed)
        stats_a = metric_fn.stats_fn(hyps_a, refs)
        stats_b = metric_fn.stats_fn(hyps_b, refs)
        deltas_arr = metric_fn.resample_fn(stats_a, idx) - metric_fn.resample_fn(stats_b, idx)
        count_b_ge_a = int(np.sum(deltas_arr <= 0))
        deltas_arr = np.sort(deltas_arr)
        lo = float(deltas_arr[int(0.025 * n_resamples)])
        hi = float(deltas_arr[min(n_resamples - 1, int(0.975 * n_resamples))])
    else:
        rng = random.Random(seed)
        deltas = []
        count_b_ge_a = 0
        for _ in range(n_resamples):
            resampled = [rng.randrange(n) for _ in range(n)]
            ha = [hyps_a[i] for i in resampled]
            hb = [hyps_b[i] for i in resampled]
            r = [refs[i] for i in resampled]
            d = metric_fn(ha, r) - metric_fn(hb, r)
            deltas.append(d)
            if d <= 0:
                count_b_ge_a += 1
        deltas.sort()
        lo = deltas[int(0.025 * n_resamples)]
        hi = deltas[min(n_resamples - 1, int(0.975 * n_resamples))]
    p_value = count_b_ge_a / n_resamples
    return {
        "delta": delta_point,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "n_resamples": n_resamples,
    }


# -------------------------------------------------------------------------------------------
# Views
# -------------------------------------------------------------------------------------------


def length_bucket_label(n_words: int) -> str:
    """Source-length-in-words bucket label (spec §8: <=10, 11-20, 21-40, 41-80, >80)."""
    if n_words <= 10:
        return "<=10"
    if n_words <= 20:
        return "11-20"
    if n_words <= 40:
        return "21-40"
    if n_words <= 80:
        return "41-80"
    return ">80"


def length_bucket_view(
    rows: list[dict[str, str]],
    pred: dict[str, str],
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """`rows`: [{id, source, reference}, ...] pooled across E1+E2+E3 (spec §8). Returns official
    BLEU/chrF per non-empty length bucket, each with a bootstrap 95% CI (`bleu_ci`/`chrf_ci`,
    spec §8: "per slice and metric" -- length buckets are a reported view alongside slices/E-sets)
    computed on that bucket's own sentences."""
    module = load_official_module()
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        label = length_bucket_label(len(row["source"].split()))
        buckets[label].append((pred.get(row["id"], ""), row["reference"]))
    result: dict[str, Any] = {}
    for label, pairs in buckets.items():
        if not pairs or label not in LENGTH_BUCKET_LABELS:
            continue
        hs = [p[0] for p in pairs]
        rs = [p[1] for p in pairs]
        scores = module.score_slice(hs, rs)
        scores["bleu_ci"] = bootstrap_ci_official(hs, rs, "bleu", n_resamples, seed)
        scores["chrf_ci"] = bootstrap_ci_official(hs, rs, "chrf", n_resamples, seed)
        result[label] = scores
    return result


# -------------------------------------------------------------------------------------------
# COMET (eval-only; isolated env via envs/comet, spec §8)
# -------------------------------------------------------------------------------------------

COMET_ENV_DIR = REPO_ROOT / "envs" / "comet"
COMET_SCRIPT = COMET_ENV_DIR / "score_comet.py"


def run_comet(
    triples: list[dict[str, str]], out_path: Path, timeout_seconds: float = 3600.0
) -> dict[str, Any]:
    """Score `triples` (each `{"src", "mt", "ref"}`) with COMET-22 (`Unbabel/wmt22-comet-da`) by
    shelling out to the isolated `envs/comet` uv project (`unbabel-comet` does not co-resolve
    with this repo's pinned torch/numpy -- see pyproject.toml). Eval-only: never used for
    checkpoint/decoding selection (spec §8, §15). Raises `RuntimeError` with the subprocess's
    stderr on failure -- never silently fabricates a score.
    """
    in_path = out_path.with_suffix(".in.json")
    in_path.write_text(json.dumps(triples), encoding="utf-8")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(COMET_ENV_DIR),
            "python",
            str(COMET_SCRIPT),
            "--in",
            str(in_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"COMET scoring failed (exit {result.returncode}):\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------------------------------
# Top-level orchestration
# -------------------------------------------------------------------------------------------


@dataclass
class EvalRunConfig:
    """Decoding + statistics config for one `run_evaluation` call. Recorded verbatim in eval.json
    ("decoding config", spec §8)."""

    beam_size: int = 5
    alpha: float = 0.6
    segment_threshold: int | None = None
    batch_size: int = 16
    n_bootstrap: int = 1000
    bootstrap_seed: int = 1234
    comet: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns `(inputs_rows, labels_rows)` for `name` in {"e1", "e2", "e3", "dev"}."""
    if name in ("e1", "e2", "e3"):
        base = REPO_ROOT / "data" / "eval" / name
    elif name == "dev":
        base = REPO_ROOT / "data" / "dev"
    else:
        raise ValueError(f"unknown split: {name!r}")
    return _read_jsonl(base / "inputs.jsonl"), _read_jsonl(base / "labels.jsonl")


def run_evaluation(
    translator: Translator,
    run_name: str,
    ckpt_name: str,
    decode_cfg: EvalRunConfig,
    out_dir: Path | None = None,
    splits: Sequence[str] = ("dev", "e1", "e2", "e3"),
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate every `splits` entry with `translator`, score it (official + sacreBLEU +
    bootstrap CIs), build the length-bucket view over E1+E2+E3, optionally run COMET, and write
    `reports/<run_name>/<ckpt_name>/eval.json` + each split's predictions (spec §8 output shape).
    """
    out_dir = out_dir or (REPO_ROOT / "reports" / run_name / ckpt_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "run": run_name,
        "checkpoint": ckpt_name,
        "decoding_config": asdict(decode_cfg),
        "provenance": provenance or {},
        "timings_seconds": {},
        "sets": {},
    }

    all_pred: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}

    for split in splits:
        t0 = time.monotonic()
        inputs, labels = load_split(split)
        label_by_id = {r["id"]: r for r in labels}
        ids = [r["id"] for r in inputs]
        sources = [r["source"] for r in inputs]

        stats_before = (
            translator.stats.n_beam,
            translator.stats.n_greedy_fallback,
            translator.stats.n_copy_fallback,
        )
        translations = translator.translate(
            sources,
            batch_size=decode_cfg.batch_size,
            beam=decode_cfg.beam_size,
            alpha=decode_cfg.alpha,
            segment_threshold=decode_cfg.segment_threshold,
        )
        stats_after = (
            translator.stats.n_beam,
            translator.stats.n_greedy_fallback,
            translator.stats.n_copy_fallback,
        )
        pred = dict(zip(ids, translations, strict=True))
        result["timings_seconds"][split] = round(time.monotonic() - t0, 3)

        pred_path = out_dir / f"{split}_predictions.json"
        pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

        gold_rows = [
            {
                "id": r["id"],
                "reference": label_by_id[r["id"]]["reference"],
                "slice": label_by_id[r["id"]].get("slice", split),
            }
            for r in inputs
        ]
        official = compute_official_metrics(pred, gold_rows)
        hyps = [pred[i] for i in ids]
        refs = [label_by_id[i]["reference"] for i in ids]
        slices = [r["slice"] for r in gold_rows]

        entry: dict[str, Any] = {
            "n": len(ids),
            "official": official,
            "official_bleu_ci": bootstrap_ci_official(
                hyps, refs, "bleu", decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
            ),
            "official_chrf_ci": bootstrap_ci_official(
                hyps, refs, "chrf", decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
            ),
            # Per-slice CIs (spec §8: "per slice and metric") -- every official dev slice
            # (seen/long/unseen_domain), and, for e1/e2/e3, the (single) E-set slice itself.
            "official_ci_by_slice": {
                "bleu": bootstrap_ci_by_group(
                    hyps, refs, slices, "bleu", decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
                ),
                "chrf": bootstrap_ci_by_group(
                    hyps, refs, slices, "chrf", decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
                ),
            },
            "sacrebleu": sacrebleu_metrics(hyps, refs),
            "fallback_counts": {
                "beam": stats_after[0] - stats_before[0],
                "greedy": stats_after[1] - stats_before[1],
                "copy": stats_after[2] - stats_before[2],
            },
        }
        if split == "dev":
            entry["overall_ci"] = bootstrap_official_overall(
                pred, gold_rows, decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
            )
        result["sets"][split] = entry
        all_pred.update(pred)
        rows_by_split[split] = [
            {"id": r["id"], "source": r["source"], "reference": label_by_id[r["id"]]["reference"]}
            for r in inputs
        ]

    combined = [
        row
        for split in ("e1", "e2", "e3")
        if split in rows_by_split
        for row in rows_by_split[split]
    ]
    if combined:
        result["length_buckets_e1_e2_e3"] = length_bucket_view(
            combined, all_pred, decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
        )

    if decode_cfg.comet:
        comet_triples = [
            {"src": row["source"], "mt": all_pred.get(row["id"], ""), "ref": row["reference"]}
            for split in rows_by_split
            for row in rows_by_split[split]
        ]
        comet_out = out_dir / "comet.json"
        try:
            result["comet"] = run_comet(comet_triples, comet_out)
        except Exception as exc:  # noqa: BLE001 - report the exact error, never fabricate a score
            result["comet_error"] = repr(exc)

    (out_dir / "eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# -------------------------------------------------------------------------------------------
# nmt.train's periodic eval hook (spec §11)
# -------------------------------------------------------------------------------------------


@dataclass
class TrainEvalConfig:
    """Config for the cheap, periodic eval hook `nmt.train.train`'s loop calls every
    `eval.eval_every` steps (spec §11). Fixed, seeded subsets (not full splits) keep this
    affordable during training; `smoke.yaml`-style configs pass small `*_n` values.
    """

    tokenizer_path: Path
    e1_n: int = 500
    e2_n: int = 300
    e3_n: int = 300
    max_len_a: float = 1.5
    max_len_b: int = 10
    no_repeat_ngram_size: int = 3
    n_samples_table: int = 20
    seed: int = 1234
    device: str = "cpu"


def build_train_eval_fn(cfg: TrainEvalConfig) -> Callable[[Any, int, Any], dict[str, Any]]:
    """Build an `nmt.train.EvalFn`: greedy BLEU/chrF (the official §8 functions, via
    `load_official_module`) on fixed seeded subsets of E1[cfg.e1_n]/E2[cfg.e2_n], the *whole*
    official dev set by slice, and E3[cfg.e3_n] (reporting only -- every E3 metric key is
    prefixed `e3_reporting_only_` so it can never be mistaken for a selection signal downstream).
    Also logs a W&B Table of `cfg.n_samples_table` fixed sample translations directly to the
    passed-in run (not through the returned dict, which stays pure float/str/int -- safe for the
    training loop's `json.dumps` into metrics.jsonl).
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(cfg.tokenizer_path))
    module = load_official_module()
    device = torch.device(cfg.device)
    rng = random.Random(cfg.seed)

    def _label_rows(split: str, n: int | None) -> list[dict[str, str]]:
        inputs, labels = load_split(split)
        label_by_id = {r["id"]: r for r in labels}
        subset = inputs if n is None or n >= len(inputs) else rng.sample(inputs, n)
        return [
            {
                "id": r["id"],
                "source": r["source"],
                "reference": label_by_id[r["id"]]["reference"],
                "slice": label_by_id[r["id"]].get("slice", split),
            }
            for r in subset
        ]

    fixed_e1 = _label_rows("e1", cfg.e1_n)
    fixed_e2 = _label_rows("e2", cfg.e2_n)
    fixed_e3 = _label_rows("e3", cfg.e3_n)
    fixed_dev = _label_rows("dev", None)  # always the whole (150-sentence) dev set
    sample_pool = fixed_dev if len(fixed_dev) >= cfg.n_samples_table else fixed_e1
    sample_rows = (
        sample_pool
        if len(sample_pool) <= cfg.n_samples_table
        else rng.sample(sample_pool, cfg.n_samples_table)
    )

    @torch.no_grad()
    def _greedy_translate(model: Any, rows: list[dict[str, str]]) -> dict[str, str]:
        if not rows:
            return {}
        ids_list = [sp.encode(r["source"], out_type=int) for r in rows]
        src, src_mask = _pad_ids(ids_list, model.cfg.pad_id, model.cfg.eos_id, device)
        hyps = greedy_decode(
            model,
            src,
            src_mask,
            model.cfg.bos_id,
            model.cfg.eos_id,
            model.cfg.pad_id,
            max_len_a=cfg.max_len_a,
            max_len_b=cfg.max_len_b,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        )
        return {
            r["id"]: _detok(sp, h.tokens, model.cfg.eos_id) for r, h in zip(rows, hyps, strict=True)
        }

    def eval_fn(model: Any, step: int, wandb_run: Any = None) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        metrics: dict[str, Any] = {}

        for name, rows in (("e1", fixed_e1), ("e2", fixed_e2)):
            pred = _greedy_translate(model, rows)
            scores = module.score_slice(
                [pred[r["id"]] for r in rows], [r["reference"] for r in rows]
            )
            metrics[f"{name}_bleu"], metrics[f"{name}_chrf"], metrics[f"{name}_n"] = (
                scores["bleu"],
                scores["chrf"],
                scores["n"],
            )

        pred_dev = _greedy_translate(model, fixed_dev)
        by_slice: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in fixed_dev:
            by_slice[r["slice"]].append((pred_dev[r["id"]], r["reference"]))
        for slice_name, pairs in by_slice.items():
            scores = module.score_slice([p[0] for p in pairs], [p[1] for p in pairs])
            metrics[f"dev_{slice_name}_bleu"] = scores["bleu"]
            metrics[f"dev_{slice_name}_chrf"] = scores["chrf"]
            metrics[f"dev_{slice_name}_n"] = scores["n"]

        pred_e3 = _greedy_translate(model, fixed_e3)
        scores_e3 = module.score_slice(
            [pred_e3[r["id"]] for r in fixed_e3], [r["reference"] for r in fixed_e3]
        )
        metrics["e3_reporting_only_bleu"] = scores_e3["bleu"]
        metrics["e3_reporting_only_chrf"] = scores_e3["chrf"]
        metrics["e3_reporting_only_n"] = scores_e3["n"]

        if wandb_run is not None and sample_rows:
            try:
                import wandb

                pred_samples = _greedy_translate(model, sample_rows)
                table = wandb.Table(columns=["id", "slice", "source", "reference", "hypothesis"])
                for r in sample_rows:
                    table.add_data(
                        r["id"], r["slice"], r["source"], r["reference"], pred_samples[r["id"]]
                    )
                wandb_run.log({"eval/sample_translations": table}, step=step)
            except Exception as exc:  # noqa: BLE001 - W&B failures must never abort training
                metrics["sample_table_error"] = repr(exc)

        if was_training:
            model.train()
        return metrics

    return eval_fn
