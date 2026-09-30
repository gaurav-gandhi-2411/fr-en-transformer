from __future__ import annotations

# Tests for nmt/data/loader.py: token-bucketed batching respects max_tokens, resuming a sampler
# yields exactly the same remaining batches, and concat augmentation respects its length cap and
# is seeded (deterministic given seed). Spec §12.
from pathlib import Path

from nmt.data.loader import BucketedSampler, ShardDataset
from tests.fixtures import build_tiny_shard_dir


def test_bucketing_respects_max_tokens(tiny_shard_dir: Path) -> None:
    dataset = ShardDataset(tiny_shard_dir, "train")
    max_tokens = 64
    sampler = BucketedSampler(dataset, max_tokens=max_tokens, seed=42, chunk_size=8)

    n_examples = 0
    for batch in sampler:
        b = batch.src.size(0)
        max_len = max(batch.src.size(1), batch.tgt_in.size(1))  # type: ignore[union-attr]
        # A batch is only split when adding the next example would exceed max_tokens (see
        # BucketedSampler._build_epoch_batches), so any single-example batch may still exceed the
        # budget on its own (an unavoidably long example) — but batches with >1 example must
        # obey it.
        if b > 1:
            assert b * max_len <= max_tokens
        n_examples += b
    assert n_examples == len(dataset)


def test_resume_yields_identical_remaining_batches(tiny_shard_dir: Path) -> None:
    dataset = ShardDataset(tiny_shard_dir, "train")

    full = BucketedSampler(dataset, max_tokens=96, seed=7, chunk_size=8, concat_prob=0.3)
    full_batches = [b.src.clone() for b in full]

    half = BucketedSampler(dataset, max_tokens=96, seed=7, chunk_size=8, concat_prob=0.3)
    n_total = len(half)
    cutoff = n_total // 2
    first_half = []
    it = iter(half)
    for _ in range(cutoff):
        first_half.append(next(it).src.clone())

    state = half.state_dict()

    resumed = BucketedSampler(dataset, max_tokens=96, seed=7, chunk_size=8, concat_prob=0.3)
    resumed.load_state_dict(state)
    resumed_batches = [b.src.clone() for b in resumed]

    remaining_full = full_batches[cutoff:]
    assert len(remaining_full) == len(resumed_batches)
    for a, b in zip(remaining_full, resumed_batches, strict=True):
        assert a.shape == b.shape
        assert (a == b).all()


def test_concat_augmentation_respects_cap_and_is_seeded(tmp_path: Path) -> None:
    shard_dir = build_tiny_shard_dir(tmp_path / "shards2", seed=1, n_train=128, n_eval=8)
    dataset = ShardDataset(shard_dir, "train")

    sampler_a = BucketedSampler(
        dataset, max_tokens=2000, seed=99, chunk_size=32, concat_prob=1.0, concat_max_len=20
    )
    batches_a = [b.src.clone() for b in sampler_a]
    for b in batches_a:
        assert b.size(1) <= 21  # concat_max_len + EOS

    sampler_b = BucketedSampler(
        dataset, max_tokens=2000, seed=99, chunk_size=32, concat_prob=1.0, concat_max_len=20
    )
    batches_b = [b.src.clone() for b in sampler_b]
    assert len(batches_a) == len(batches_b)
    for a, b in zip(batches_a, batches_b, strict=True):
        assert a.shape == b.shape
        assert (a == b).all()


def test_no_augmentation_when_concat_prob_zero(tiny_shard_dir: Path) -> None:
    dataset = ShardDataset(tiny_shard_dir, "train")
    sampler = BucketedSampler(dataset, max_tokens=1000, seed=1, chunk_size=64, concat_prob=0.0)
    for batch in sampler:
        # With concat off, every source row's non-pad length is exactly the sum of one original
        # example's tokens + 1 EOS — bounded by the synthetic generator's max_len=12 + 1.
        assert batch.src.size(1) <= 13
