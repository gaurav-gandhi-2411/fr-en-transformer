from __future__ import annotations

# Tests for scripts/comet_sanity.py helpers: the shuffled-reference control must move every
# reference off its own sentence, and the dev sample must be stratified and deterministic.
from scripts.comet_sanity import STRATA, derange, sample_ids


def test_derange_has_no_fixed_points_and_keeps_items() -> None:
    items = [f"ref{i}" for i in range(50)]
    out = derange(items, seed=1234)
    assert sorted(out) == sorted(items)
    assert all(a != b for a, b in zip(items, out, strict=True))


def test_sample_ids_is_stratified_and_deterministic() -> None:
    labels = [{"id": f"{s}_{i}", "slice": s} for s in STRATA for i in range(60)]
    ids = sample_ids(labels, seed=1234)
    assert ids == sample_ids(labels, seed=1234)
    for slice_name, k in STRATA.items():
        assert sum(i.startswith(slice_name) for i in ids) == k
