from __future__ import annotations

# Tests for the concat-augmentation epoch plan in nmt/data/loader.py. Bug: pairs were joined at
# materialization, AFTER bucketing on unjoined lengths, and the whole batch was then padded to its
# longest joined row, so padded micro-batches reached 131k-170k tokens against a 4k-8k budget
# (reports/epoch_accounting.json, pre-fix). Now the plan (who is joined, with whom) is drawn per
# (seed, epoch) before bucketing and bucketing uses the joined lengths.
from pathlib import Path

import numpy as np
import pytest

from nmt.data.loader import BucketedSampler, ShardDataset, batch_token_count
from tests.fixtures import build_tiny_shard_dir


def _old_build_epoch_batches(
    dataset: ShardDataset, max_tokens: int, seed: int, epoch: int, chunk_size: int, shuffle: bool
) -> list[list[int]]:
    """Verbatim copy of the pre-plan `BucketedSampler._build_epoch_batches` (reference oracle)."""

    def example_len(i: int) -> int:
        src_len = dataset.src_len(i) + 1
        tgt_len = (dataset.tgt_len(i) + 1) if dataset.has_tgt else 0
        return max(src_len, tgt_len)

    order_rng = np.random.default_rng((seed, epoch))
    n = len(dataset)
    perm = order_rng.permutation(n) if shuffle else np.arange(n)
    batches: list[list[int]] = []
    for start in range(0, n, chunk_size):
        chunk = perm[start : start + chunk_size]
        lens = np.array([example_len(int(i)) for i in chunk])
        chunk_sorted = chunk[np.argsort(lens, kind="stable")]
        cur: list[int] = []
        cur_max = 0
        for i in chunk_sorted:
            length = example_len(int(i))
            new_max = max(cur_max, length)
            if cur and (len(cur) + 1) * new_max > max_tokens:
                batches.append(cur)
                cur = [int(i)]
                cur_max = length
            else:
                cur.append(int(i))
                cur_max = new_max
        if cur:
            batches.append(cur)
    if shuffle and batches:
        batch_order = order_rng.permutation(len(batches))
        batches = [batches[j] for j in batch_order]
    return batches


@pytest.fixture
def mid_shard_dir(tmp_path: Path) -> Path:
    return build_tiny_shard_dir(tmp_path / "mid", seed=3, n_train=3000, n_eval=8)


@pytest.mark.parametrize("shuffle", [True, False])
def test_concat_off_batches_identical_to_old_implementation(
    mid_shard_dir: Path, shuffle: bool
) -> None:
    dataset = ShardDataset(mid_shard_dir, "train")
    sampler = BucketedSampler(
        dataset, max_tokens=96, seed=42, chunk_size=64, concat_prob=0.0, shuffle=shuffle
    )
    for epoch in range(3):
        sampler.set_epoch(epoch)
        expected = _old_build_epoch_batches(dataset, 96, 42, epoch, 64, shuffle)
        assert sampler._batches == expected
    # And the materialized tensors carry exactly the un-joined examples, in that order.
    sampler.set_epoch(0)
    first = next(iter(sampler))
    src0, _, ex_id0 = dataset.get(sampler._batches[0][0])
    assert first.ids[0] == ex_id0
    assert first.src[0, : len(src0)].tolist() == src0.astype(np.int64).tolist()


@pytest.mark.parametrize("concat_prob", [0.0, 0.15, 0.5, 1.0])
def test_padded_batches_respect_budget_with_concat(mid_shard_dir: Path, concat_prob: float) -> None:
    dataset = ShardDataset(mid_shard_dir, "train")
    max_tokens, cap = 96, 40  # cap well below 4 x 12 so truncation is exercised
    sampler = BucketedSampler(
        dataset,
        max_tokens=max_tokens,
        seed=5,
        chunk_size=64,
        concat_prob=concat_prob,
        concat_max_len=cap,
    )
    n_examples = 0
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for batch in sampler:
            assert batch.tgt_in is not None
            n_examples += batch.src.size(0)
            # The no-concat bound: each padded side <= max_tokens, so src + tgt_in <= 2 x budget
            # (a lone example longer than the budget is the only allowed exception).
            assert batch.src.numel() <= max_tokens or batch.src.size(0) == 1
            assert batch.tgt_in.numel() <= max_tokens or batch.tgt_in.size(0) == 1
            assert batch_token_count(batch) <= 2 * max_tokens or batch.src.size(0) == 1
            assert batch.src.size(1) <= cap + 1
    assert n_examples == 2 * len(dataset)  # every example is the primary of exactly one row


def test_concat_rate_and_joined_lengths(mid_shard_dir: Path) -> None:
    dataset = ShardDataset(mid_shard_dir, "train")
    cap = 30
    sampler = BucketedSampler(
        dataset, max_tokens=96, seed=5, chunk_size=64, concat_prob=0.15, concat_max_len=cap
    )
    joined = sampler._concat_k >= 2
    assert abs(joined.mean() - 0.15) < 0.02  # n=3000: sd of the rate is ~0.0065
    assert set(np.unique(sampler._concat_k[joined])) == {2, 3, 4}

    id_to_index = {dataset.get(i)[2]: i for i in range(len(dataset))}
    n_checked = 0
    for batch in sampler:
        for row, ex_id in enumerate(batch.ids):
            i = id_to_index[ex_id]
            k = int(sampler._concat_k[i])
            parts = [i] + [int(j) for j in sampler._partners[i, : max(k - 1, 0)]]
            want = min(sum(dataset.src_len(j) for j in parts), cap) + 1  # + EOS
            assert int((batch.src[row] != 0).sum()) == want
            n_checked += k >= 2
    assert n_checked == int(joined.sum())


def test_plan_is_pure_function_of_seed_and_epoch(mid_shard_dir: Path) -> None:
    dataset = ShardDataset(mid_shard_dir, "train")
    a = BucketedSampler(dataset, max_tokens=96, seed=9, chunk_size=64, concat_prob=0.3)
    b = BucketedSampler(dataset, max_tokens=96, seed=9, chunk_size=64, concat_prob=0.3)
    a.set_epoch(4)
    b.set_epoch(4)
    assert (a._concat_k == b._concat_k).all() and (a._partners == b._partners).all()
    assert a._batches == b._batches
    b.set_epoch(5)
    assert not (a._concat_k == b._concat_k).all()


def test_resume_mid_epoch_after_epoch_rollover_with_concat(mid_shard_dir: Path) -> None:
    dataset = ShardDataset(mid_shard_dir, "train")

    def make() -> BucketedSampler:
        return BucketedSampler(dataset, max_tokens=96, seed=7, chunk_size=64, concat_prob=0.3)

    full = make()
    full.set_epoch(2)
    full_batches = [b.src.clone() for b in full]

    half = make()
    half.set_epoch(2)
    it = iter(half)
    cutoff = len(half) // 3
    for _ in range(cutoff):
        next(it)
    state = half.state_dict()
    assert set(state) == {"epoch", "batch_cursor"}

    resumed = make()
    resumed.load_state_dict(state)
    rest = [b.src.clone() for b in resumed]
    assert len(rest) == len(full_batches) - cutoff
    for x, y in zip(full_batches[cutoff:], rest, strict=True):
        assert x.shape == y.shape and (x == y).all()


def test_load_state_dict_accepts_legacy_aug_rng_key(mid_shard_dir: Path) -> None:
    dataset = ShardDataset(mid_shard_dir, "train")
    s = BucketedSampler(dataset, max_tokens=96, seed=7, chunk_size=64, concat_prob=0.3)
    s.load_state_dict({"epoch": 1, "batch_cursor": 2, "aug_rng_state": {"legacy": True}})
    assert (s.epoch, s.batch_cursor) == (1, 2)
