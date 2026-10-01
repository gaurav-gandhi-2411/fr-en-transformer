from __future__ import annotations

# Fast tests for scripts/audit_leakage.py's bucketing/aggregation helpers on tiny hand-built
# records (the full re-derivation over the real datasets is run by the script itself and asserted
# against data_manifest.json there; it is deliberately not part of the unit suite).
import pytest

from scripts.audit_leakage import (
    SetMatcher,
    breakdown_exclusive,
    breakdown_non_exclusive,
    breakdown_word_buckets,
    coarse_matches,
    first_match,
    fr_word_count,
    merge_set_matches,
    side_label,
    word_bucket,
)


def _rec(match_type: str, bucket: str, **sets: tuple[str, str]) -> dict:
    return {
        "match_type": match_type,
        "bucket": bucket,
        "sets": {k: {"match_type": v[0], "side": v[1]} for k, v in sets.items()},
    }


def test_word_bucket_boundaries() -> None:
    assert [word_bucket(n) for n in (1, 3, 4, 10, 11, 200)] == [
        "1-3",
        "1-3",
        "4-10",
        "4-10",
        ">10",
        ">10",
    ]
    assert fr_word_count("Oh,  non ! ") == 3
    assert fr_word_count("Oh, non !") == 3


def test_side_label() -> None:
    assert side_label(True, True) == "both"
    assert side_label(True, False) == "fr"
    assert side_label(False, True) == "en"
    assert side_label(False, False) is None


def test_merge_set_matches_exact_wins_and_sides_union() -> None:
    a = {"match_type": "near_dup", "side": "fr"}
    b = {"match_type": "exact", "side": "en"}
    assert merge_set_matches([a, None, b]) == {"match_type": "exact", "side": "both"}
    assert merge_set_matches([a, None]) == a
    assert merge_set_matches([None, None]) is None


def test_coarse_matches_groups_dev_src_and_dev_ref() -> None:
    fine = {
        "dev_src": {"match_type": "near_dup", "side": "fr"},
        "dev_ref": {"match_type": "near_dup", "side": "en"},
        "test_src": None,
        "e1": None,
        "e3": {"match_type": "exact", "side": "fr"},
    }
    out = coarse_matches(fine)
    assert set(out) == {"dev", "e3"}
    assert out["dev"] == {"match_type": "near_dup", "side": "both"}


def test_first_match_follows_priority() -> None:
    assert first_match(["e3", "e1", "test"]) == "test"
    assert first_match(["e3", "e1"]) == "e1"
    with pytest.raises(ValueError):
        first_match(["unknown"])


def test_non_exclusive_counts_a_pair_in_every_set_it_matches() -> None:
    recs = [
        _rec("exact", "1-3", e1=("exact", "fr"), e3=("near_dup", "en")),
        _rec("near_dup", "1-3", e3=("near_dup", "both")),
    ]
    out = breakdown_non_exclusive(recs)
    assert out["e1"]["total"] == 1 and out["e1"]["by_match_type"]["exact"] == 1
    assert out["e3"]["total"] == 2
    assert out["e3"]["by_side"] == {"fr": 0, "en": 1, "both": 1}
    assert out["e2"]["total"] == 0


def test_exclusive_attributes_each_pair_once_and_sums_to_total() -> None:
    recs = [
        _rec("exact", "1-3", e1=("exact", "fr"), e3=("near_dup", "en")),  # -> e1
        _rec("near_dup", "4-10", dev=("near_dup", "en"), e3=("near_dup", "fr")),  # -> dev
        _rec("near_dup", "1-3", e3=("near_dup", "both")),  # -> e3
    ]
    out = breakdown_exclusive(recs)
    assert sum(c["total"] for c in out.values()) == len(recs)
    assert out["e1"]["by_match_type"] == {"exact": 1, "near_dup": 0}
    assert out["dev"]["by_side"]["en"] == 1
    assert out["e3"]["total"] == 1
    assert sum(c["by_match_type"]["exact"] for c in out.values()) == 1


def test_word_bucket_breakdown_and_near_dup_share() -> None:
    recs = [
        _rec("near_dup", "1-3"),
        _rec("near_dup", "1-3"),
        _rec("near_dup", "4-10"),
        _rec("exact", ">10"),
    ]
    pool = {"1-3": 10, "4-10": 20, ">10": 70}
    out = breakdown_word_buckets(recs, pool)
    assert out["by_bucket_and_match_type"]["1-3"] == {"exact": 0, "near_dup": 2, "total": 2}
    assert out["share_of_near_dup_removals_that_are_1_3_words"] == pytest.approx(2 / 3)
    assert out["share_of_all_removals_that_are_1_3_words"] == pytest.approx(2 / 4)
    assert out["removal_rate_per_bucket"]["1-3"] == pytest.approx(0.2)
    assert out["pool_bucket_share"][">10"] == pytest.approx(0.7)


def test_set_matcher_exact_vs_near_dup_and_targets() -> None:
    m = SetMatcher(["Oui.", "-- Non !", "Hello there"])
    assert m.match("Oui.", "zzz") == {"match_type": "exact", "side": "fr"}
    assert m.match("oui !", "x y") == {"match_type": "near_dup", "side": "fr"}
    assert m.match("Non", "hello there!") == {"match_type": "near_dup", "side": "both"}
    assert m.match("Bonjour", "Hi") is None
    assert m.targets_for("non.") == ["-- Non !"]
    assert m.targets_for("Oui.") == ["Oui."]
