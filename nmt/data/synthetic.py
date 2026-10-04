from __future__ import annotations

# Synthetic copy-task shard generator: writes shard npz files that conform exactly to the real
# shard format without requiring prepared/tokenized data. Used by `nmt.train --synthetic` for
# CPU dry runs, and by tests/fixtures.py for fast, deterministic unit tests against
# nmt/data/loader.py and nmt/model/transformer.py.
from pathlib import Path

import numpy as np

# Content token ids start after the 4 special ids (pad=0, unk=1, bos=2, eos=3) —
# synthetic examples must never accidentally emit a special id as "real" content, or the
# loader's BOS/EOS bookkeeping would be ambiguous.
_FIRST_CONTENT_ID = 4


def _random_example(
    rng: np.random.Generator, vocab_size: int, min_len: int, max_len: int
) -> np.ndarray:
    length = int(rng.integers(min_len, max_len + 1))
    return rng.integers(_FIRST_CONTENT_ID, vocab_size, size=length).astype(np.uint16)


def write_synthetic_split(
    split_dir: Path,
    n_examples: int,
    vocab_size: int,
    seed: int,
    min_len: int = 5,
    max_len: int = 40,
    copy_task: bool = True,
    has_tgt: bool = True,
) -> None:
    """Write a single `shard_00000.npz` under `split_dir` with `n_examples` random pairs.

    `copy_task=True` sets target == source (an easy, learnable task for smoke-testing that loss
    decreases). Matches the shard contract: `src`/`tgt` uint16 flat arrays, `src_off`/`tgt_off`
    int64 offset arrays of length n+1, `ids` a unicode array — `tgt`/`tgt_off` are omitted
    entirely when `has_tgt=False`, mirroring the real `test` split.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    srcs: list[np.ndarray] = []
    src_off = [0]
    tgts: list[np.ndarray] = []
    tgt_off = [0]
    ids = []
    for i in range(n_examples):
        s = _random_example(rng, vocab_size, min_len, max_len)
        srcs.append(s)
        src_off.append(src_off[-1] + len(s))
        if has_tgt:
            t = s.copy() if copy_task else _random_example(rng, vocab_size, min_len, max_len)
            tgts.append(t)
            tgt_off.append(tgt_off[-1] + len(t))
        ids.append(f"synth-{i:06d}")

    payload: dict[str, np.ndarray] = {
        "src": np.concatenate(srcs).astype(np.uint16) if srcs else np.zeros(0, dtype=np.uint16),
        "src_off": np.array(src_off, dtype=np.int64),
        "ids": np.array(ids, dtype="<U16"),
    }
    if has_tgt:
        payload["tgt"] = (
            np.concatenate(tgts).astype(np.uint16) if tgts else np.zeros(0, dtype=np.uint16)
        )
        payload["tgt_off"] = np.array(tgt_off, dtype=np.int64)
    np.savez(split_dir / "shard_00000.npz", **payload)


def write_synthetic_shard_dir(
    root: Path,
    vocab_size: int,
    seed: int,
    n_train: int = 512,
    n_eval: int = 64,
    min_len: int = 5,
    max_len: int = 40,
) -> Path:
    """Build a full synthetic shard tree (train + e1 splits) under `root`. Returns `root`."""
    write_synthetic_split(root / "train", n_train, vocab_size, seed, min_len, max_len)
    write_synthetic_split(root / "e1", n_eval, vocab_size, seed + 1, min_len, max_len)
    return root
