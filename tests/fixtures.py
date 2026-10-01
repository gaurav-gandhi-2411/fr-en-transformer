from __future__ import annotations

# Shared, non-pytest test helpers: builds tiny synthetic shard directories via
# `nmt.data.synthetic.write_synthetic_shard_dir`, matching the P2 shard contract exactly
# (PLAN.md "Interface contracts") so tests run without real prepared/tokenized data — P2 hasn't
# been built yet. The pytest fixture wrapper lives in `tests/conftest.py` so it's
# auto-discovered; this module holds the plain function so non-fixture callers (e.g. a test
# building several shard dirs with different sizes) can call it directly too. Spec §12.
from pathlib import Path

from nmt.data.synthetic import write_synthetic_shard_dir

TINY_VOCAB_SIZE = 64


def build_tiny_shard_dir(root: Path, seed: int = 0, n_train: int = 64, n_eval: int = 16) -> Path:
    """Build a tiny synthetic shard tree (train + e1 splits, copy task) under `root`."""
    return write_synthetic_shard_dir(
        root,
        vocab_size=TINY_VOCAB_SIZE,
        seed=seed,
        n_train=n_train,
        n_eval=n_eval,
        min_len=3,
        max_len=12,
    )
