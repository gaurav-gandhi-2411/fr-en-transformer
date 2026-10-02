from __future__ import annotations

# Tests for nmt/analysis.py's per-group failure-mode rates and source-rarity buckets (the
# reports/final tables). Hand-built features, no tokenizer needed. Spec §10.
import pytest

from nmt.analysis import (
    SentenceFeatures,
    failure_mode_rates,
    rarity_bucket_chrf,
    rarity_bucket_edges,
)


def _feat(
    *,
    ratio: float = 1.0,
    rep: float = 0.0,
    trunc: bool = False,
    copy: float = 0.0,
    rarity: float = 10.0,
    chrf: float = 50.0,
) -> SentenceFeatures:
    return SentenceFeatures(
        id="x",
        domain="e1",
        length_ratio=ratio,
        repetition_rate=rep,
        truncated=trunc,
        untranslated_copy_rate=copy,
        src_rarity_mean=rarity,
        src_rarity_min=1.0,
        src_byte_fallback_rate=0.0,
        proper_noun_copy_accuracy=None,
        dialogue_punct_density=0.0,
        chrf=chrf,
    )


def test_failure_mode_rates_counts_each_mode() -> None:
    feats = [
        _feat(),
        _feat(rep=0.2),
        _feat(ratio=0.3, trunc=True),
        _feat(copy=0.5),
        _feat(ratio=2.0),
    ]
    r = failure_mode_rates(feats)
    assert r["n"] == 5
    assert r["repetition_rate"] == pytest.approx(1 / 5)
    assert r["truncation_rate"] == pytest.approx(1 / 5)
    assert r["untranslated_copy_rate"] == pytest.approx(1 / 5)
    assert r["overlong_rate"] == pytest.approx(1 / 5)
    assert r["length_ratio_mean"] == pytest.approx((1 + 1 + 0.3 + 1 + 2) / 5)
    assert r["length_ratio_median"] == pytest.approx(1.0)


def test_failure_mode_rates_empty_group_is_none_not_zero() -> None:
    r = failure_mode_rates([])
    assert r["n"] == 0
    assert r["repetition_rate"] is None and r["length_ratio_mean"] is None


def test_failure_mode_rates_thresholds_are_strict() -> None:
    # exactly 0.1 copy share and exactly 1.5 ratio are NOT failures (strict inequalities)
    r = failure_mode_rates([_feat(copy=0.1, ratio=1.5)])
    assert r["untranslated_copy_rate"] == 0.0 and r["overlong_rate"] == 0.0


def test_rarity_buckets_order_common_first_and_cover_every_sentence() -> None:
    feats = [_feat(rarity=float(v), chrf=float(v)) for v in range(1, 11)]
    edges = rarity_bucket_edges([f.src_rarity_mean for f in feats], n_buckets=5)
    assert len(edges) == 4
    buckets = rarity_bucket_chrf(feats, edges)
    assert [b["bucket"] for b in buckets] == [1, 2, 3, 4, 5]
    assert sum(b["n"] for b in buckets) == 10
    # frequency (and here chrF) falls as the bucket gets rarer
    chrfs = [b["chrf"] for b in buckets]
    assert chrfs == sorted(chrfs, reverse=True)
    assert buckets[0]["mean_src_freq"] > buckets[-1]["mean_src_freq"]


def test_rarity_bucket_with_no_members_is_none() -> None:
    buckets = rarity_bucket_chrf([_feat(rarity=1.0)], edges=[5.0, 10.0])
    assert buckets[-1]["n"] == 1 and buckets[0]["n"] == 0 and buckets[0]["chrf"] is None
