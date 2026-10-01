from __future__ import annotations

# Pytest fixtures shared across the test suite. Spec §12.
from pathlib import Path

import pytest

from tests.fixtures import TINY_VOCAB_SIZE, build_tiny_shard_dir

__all__ = ["TINY_VOCAB_SIZE"]


@pytest.fixture
def tiny_shard_dir(tmp_path: Path) -> Path:
    """A tiny synthetic shard tree (train + e1 splits, vocab_size=64) under tmp_path."""
    return build_tiny_shard_dir(tmp_path / "shards")
