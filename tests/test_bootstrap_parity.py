from __future__ import annotations

# Parity tests for the cached-sufficient-statistics bootstrap (nmt/evaluate.py, nmt/compare.py):
# (a) real dev set + committed smoke predictions, (b) a seeded synthetic set with edge cases
# (empty hyp, single word, very long, unicode, repeated n-grams). Checks that the stats path's
# full-set point estimates equal official/score.py within 1e-9, equal the identity-index resample,
# and equal a FROZEN copy of the pre-refactor implementation (the oracle below); that paired
# bootstrap really reuses one index matrix for A and B; and that seeding is deterministic.
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

import nmt.evaluate as ev
from nmt.evaluate import (
    _bleu_resample,
    _bleu_sentence_stats,
    _chrf_resample,
    _chrf_sentence_stats,
    _official_metric_fn,
    bootstrap_ci_by_group,
    bootstrap_ci_official,
    bootstrap_official_overall,
    compute_official_metrics,
    load_official_module,
    paired_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9
PRED_PATH = REPO_ROOT / "reports" / "smoke_tuned" / "eval" / "export" / "dev_predictions.json"

# -------------------------------------------------------------------------------------------
# Frozen pre-refactor reference oracle (copied from nmt/evaluate.py @ ba66d63; do not "improve").
# The point estimate here is the OFFICIAL string-based metric; resampling is the older numpy gather
# with `_old_bleu_from_aggregated`. Everything is independent of the new scalar point helpers.
# -------------------------------------------------------------------------------------------


def _old_bleu_from_aggregated(match, total, hyp_len, ref_len):  # type: ignore[no-untyped-def]
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
        safe_precs = np.where(valid[..., None], precs, 1.0)
        geo = np.exp(np.mean(np.log(safe_precs), axis=-1))
        safe_hyp_len = np.where(hyp_len > 0, hyp_len, 1.0)
        bp = np.where(hyp_len > ref_len, 1.0, np.exp(1.0 - ref_len / safe_hyp_len))
    return np.where(valid, 100.0 * bp * geo, 0.0)


def _old_resample(metric: str, stats, idx):  # type: ignore[no-untyped-def]
    if metric == "chrf":
        return stats[idx].mean(axis=1)
    return _old_bleu_from_aggregated(
        stats.match[idx].sum(axis=1),
        stats.total[idx].sum(axis=1),
        stats.hyp_len[idx].sum(axis=1),
        stats.ref_len[idx].sum(axis=1),
    )


def _old_point(metric: str, hyps: list[str], refs: list[str]) -> float:
    m = load_official_module()
    if metric == "bleu":
        return m.corpus_bleu(hyps, refs)
    return float(np.array([m.chrf_sentence(h, r) for h, r in zip(hyps, refs, strict=True)]).mean())


def _old_stats(metric: str, hyps: list[str], refs: list[str]):  # type: ignore[no-untyped-def]
    return (_bleu_sentence_stats if metric == "bleu" else _chrf_sentence_stats)(hyps, refs)


def _old_bootstrap_ci(metric, hyps, refs, n_resamples, seed):  # type: ignore[no-untyped-def]
    n = len(refs)
    point = _old_point(metric, hyps, refs)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    s = np.sort(_old_resample(metric, _old_stats(metric, hyps, refs), idx))
    lo = min(float(s[int(0.025 * n_resamples)]), point)
    hi = max(float(s[min(n_resamples - 1, int(0.975 * n_resamples))]), point)
    return {"point": point, "ci_low": lo, "ci_high": hi}


def _old_paired(metric, ha, hb, refs, n_resamples, seed):  # type: ignore[no-untyped-def]
    n = len(refs)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    d = _old_resample(metric, _old_stats(metric, ha, refs), idx) - _old_resample(
        metric, _old_stats(metric, hb, refs), idx
    )
    p = int(np.sum(d <= 0)) / n_resamples
    d = np.sort(d)
    return {
        "delta": _old_point(metric, ha, refs) - _old_point(metric, hb, refs),
        "ci_low": float(d[int(0.025 * n_resamples)]),
        "ci_high": float(d[min(n_resamples - 1, int(0.975 * n_resamples))]),
        "p_value": p,
    }


def _naive_samples(metric: str, hyps, refs, idx):  # type: ignore[no-untyped-def]
    """Slowest possible oracle: official `score_slice` on every resampled string list."""
    m = load_official_module()
    return np.array(
        [
            m.score_slice([hyps[i] for i in row], [refs[i] for i in row])[metric]
            for row in idx.tolist()
        ]
    )


# -------------------------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------------------------


def _dev_real() -> tuple[list[str], list[str], list[str]]:
    rows = [
        json.loads(x)
        for x in (REPO_ROOT / "data" / "dev" / "labels.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    pred = json.loads(PRED_PATH.read_text("utf-8"))
    return (
        [pred.get(r["id"], "") for r in rows],
        [r["reference"] for r in rows],
        [r.get("slice", "unspecified") for r in rows],
    )


def _synthetic(n: int = 120, seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    vocab = [
        "the",
        "a",
        "cat",
        "chat",
        "été",
        "naïve",
        "Ünï",
        "日本語",
        "x",
        "don't",
        "!",
        ",",
        ".",
    ]
    refs: list[str] = []
    hyps: list[str] = []
    for _i in range(n):
        ref = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 25)))
        hyp = " ".join(w if rng.random() < 0.6 else rng.choice(vocab) for w in ref.split())
        refs.append(ref)
        hyps.append(hyp)
    # Edge cases pinned at fixed positions.
    refs[0], hyps[0] = "a b c d e f", ""  # empty hypothesis
    refs[1], hyps[1] = "word", "word"  # single word
    refs[2], hyps[2] = " ".join(rng.choice(vocab) for _ in range(600)), "the " * 700  # very long
    refs[3], hyps[3] = "日本語 été naïve Ünï", "日本語 été naïve ÜNÏ"  # unicode + case folding
    refs[4], hyps[4] = "la la la la la la", "la la la la la la la la"  # repeated n-grams
    refs[5], hyps[5] = "", ""  # both empty
    refs[6], hyps[6] = "", "spurious"  # empty reference
    slices = ["unseen_domain" if i % 3 == 0 else "seen" for i in range(n)]
    return hyps, refs, slices


DATASETS = {"dev_real": _dev_real, "synthetic": _synthetic}


@pytest.fixture(params=sorted(DATASETS))
def data(request: pytest.FixtureRequest) -> tuple[list[str], list[str], list[str]]:
    return DATASETS[request.param]()


# -------------------------------------------------------------------------------------------
# (a)/(b) point-estimate parity
# -------------------------------------------------------------------------------------------


def test_stats_point_equals_official_score_slice(data) -> None:  # type: ignore[no-untyped-def]
    hyps, refs, _ = data
    official = load_official_module().score_slice(hyps, refs)
    for metric in ("bleu", "chrf"):
        fn = _official_metric_fn(metric)
        from_stats = fn.point_from_stats_fn(fn.stats_fn(hyps, refs))
        assert abs(from_stats - official[metric]) <= TOL
        assert abs(fn(hyps, refs) - official[metric]) <= TOL


def test_identity_index_resample_equals_point(data) -> None:  # type: ignore[no-untyped-def]
    hyps, refs, _ = data
    ident = np.arange(len(refs))[None, :]
    official = load_official_module().score_slice(hyps, refs)
    assert abs(_bleu_resample(_bleu_sentence_stats(hyps, refs), ident)[0] - official["bleu"]) <= TOL
    assert abs(_chrf_resample(_chrf_sentence_stats(hyps, refs), ident)[0] - official["chrf"]) <= TOL


def test_new_bootstrap_ci_equals_frozen_old_implementation(data) -> None:  # type: ignore[no-untyped-def]
    hyps, refs, _ = data
    for metric in ("bleu", "chrf"):
        new = bootstrap_ci_official(hyps, refs, metric, n_resamples=200, seed=1234)
        old = _old_bootstrap_ci(metric, hyps, refs, 200, 1234)
        for k in ("point", "ci_low", "ci_high"):
            assert abs(new[k] - old[k]) <= TOL, (metric, k)


def test_new_paired_equals_frozen_old_implementation(data) -> None:  # type: ignore[no-untyped-def]
    hyps, refs, _ = data
    rng = random.Random(3)
    hyps_b = [h if rng.random() < 0.7 else h.split(" ")[0] if h else "x" for h in hyps]
    for metric in ("bleu", "chrf"):
        new = paired_bootstrap(hyps, hyps_b, refs, _official_metric_fn(metric), 200, 1234)
        old = _old_paired(metric, hyps, hyps_b, refs, 200, 1234)
        assert new["p_value"] == old["p_value"]
        for k in ("delta", "ci_low", "ci_high"):
            assert abs(new[k] - old[k]) <= TOL, (metric, k)


def test_resample_equals_naive_official_recomputation_same_indices() -> None:
    hyps, refs, _ = _synthetic(n=40)
    idx = np.random.default_rng(1234).integers(0, 40, size=(15, 40))
    assert (
        np.max(
            np.abs(_bleu_resample(_bleu_sentence_stats(hyps, refs), idx))
            - _naive_samples("bleu", hyps, refs, idx)
        )
        <= TOL
    )
    assert (
        np.max(
            np.abs(_chrf_resample(_chrf_sentence_stats(hyps, refs), idx))
            - _naive_samples("chrf", hyps, refs, idx)
        )
        <= TOL
    )


def test_overall_point_and_ci_match_official_and_frozen_old(data) -> None:  # type: ignore[no-untyped-def]
    hyps, refs, slices = data
    ids = [f"id{i}" for i in range(len(refs))]
    pred = dict(zip(ids, hyps, strict=True))
    gold = [
        {"id": i, "reference": r, "slice": s} for i, r, s in zip(ids, refs, slices, strict=True)
    ]
    got = bootstrap_official_overall(pred, gold, n_resamples=200, seed=1234)
    assert abs(got["point"] - compute_official_metrics(pred, gold)["OVERALL"]) <= TOL
    # Frozen-old: same idx, official-string point, old aggregate formula.
    bs, cv = _bleu_sentence_stats(hyps, refs), _chrf_sentence_stats(hyps, refs)
    unseen = np.array([s == "unseen_domain" for s in slices])
    idx = np.random.default_rng(1234).integers(0, len(refs), size=(200, len(refs)))
    g, um = cv[idx], unseen[idx]
    cnt = um.sum(axis=1)
    cu = np.where(cnt > 0, (g * um).sum(axis=1) / np.where(cnt == 0, 1, cnt), g.mean(axis=1))
    o = np.sort(0.4 * _old_resample("bleu", bs, idx) + 0.4 * g.mean(axis=1) + 0.2 * cu)
    assert abs(got["ci_low"] - float(o[5])) <= TOL
    assert abs(got["ci_high"] - float(o[min(199, int(0.975 * 200))])) <= TOL


def test_by_group_points_equal_official_by_slice(data) -> None:  # type: ignore[no-untyped-def]
    hyps, refs, slices = data
    ids = [f"id{i}" for i in range(len(refs))]
    gold = [
        {"id": i, "reference": r, "slice": s} for i, r, s in zip(ids, refs, slices, strict=True)
    ]
    by_slice = compute_official_metrics(dict(zip(ids, hyps, strict=True)), gold)["by_slice"]
    for metric in ("bleu", "chrf"):
        got = bootstrap_ci_by_group(hyps, refs, slices, metric, n_resamples=50, seed=7)
        for g, ci in got.items():
            assert abs(ci["point"] - by_slice[g][metric]) <= TOL


# -------------------------------------------------------------------------------------------
# (c) pairing
# -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["bleu", "chrf"])
def test_paired_bootstrap_uses_one_index_matrix_for_both_systems(
    metric: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hyps_a, refs, _ = _synthetic(n=60)
    hyps_b = [h.upper() if i % 2 else h for i, h in enumerate(hyps_a)]
    calls: list[np.ndarray] = []
    real = ev._resample_indices

    def recording(n: int, n_resamples: int, seed: int) -> np.ndarray:
        calls.append(real(n, n_resamples, seed))
        return calls[-1]

    monkeypatch.setattr(ev, "_resample_indices", recording)
    fn = _official_metric_fn(metric)
    got = paired_bootstrap(hyps_a, hyps_b, refs, fn, n_resamples=100, seed=5)
    assert len(calls) == 1  # one matrix drawn; A and B cannot have been resampled independently
    idx = calls[0]
    expected = fn.resample_fn(fn.stats_fn(hyps_a, refs), idx) - fn.resample_fn(
        fn.stats_fn(hyps_b, refs), idx
    )
    assert got["p_value"] == int(np.sum(expected <= 0)) / 100
    s = np.sort(expected)
    assert got["ci_low"] == float(s[2]) and got["ci_high"] == float(s[97])


def test_paired_identical_systems_give_exactly_zero_delta_everywhere() -> None:
    hyps, refs, _ = _synthetic(n=60)
    for metric in ("bleu", "chrf"):
        r = paired_bootstrap(hyps, hyps, refs, _official_metric_fn(metric), 100, 9)
        assert (r["delta"], r["ci_low"], r["ci_high"], r["p_value"]) == (0.0, 0.0, 0.0, 1.0)


def test_objective_bootstrap_uses_shared_indices_and_matches_selection_point() -> None:
    from nmt.compare import paired_bootstrap_objective

    h, r, _ = _synthetic(n=50)
    h1, h2, r1, r2 = h[:30], h[30:], r[:30], r[30:]
    b1 = [x.upper() for x in h1]
    out = paired_bootstrap_objective((h1, h2), (b1, h2), (r1, r2), n_bootstrap=100, seed=1)
    m = load_official_module()
    comb = m.score_slice(h1 + h2, r1 + r2)
    e1 = m.score_slice(h1, r1)
    want = 0.4 * comb["bleu"] + 0.4 * comb["chrf"] + 0.2 * e1["chrf"]
    assert abs(out["a"]["objective"] - want) <= TOL
    assert out["ci_low"] <= out["ci_high"]


# -------------------------------------------------------------------------------------------
# (d) determinism
# -------------------------------------------------------------------------------------------


def test_same_seed_identical_and_different_seed_differs() -> None:
    hyps, refs, _ = _synthetic(n=80)
    a = bootstrap_ci_official(hyps, refs, "bleu", 300, 11)
    b = bootstrap_ci_official(hyps, refs, "bleu", 300, 11)
    c = bootstrap_ci_official(hyps, refs, "bleu", 300, 12)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])
    assert a["point"] == c["point"]
    assert not np.array_equal(ev._resample_indices(80, 10, 11), ev._resample_indices(80, 10, 12))
    assert math.isfinite(a["ci_low"])
