from __future__ import annotations

# Token-bucketed batching over pre-tokenized shards, concatenation augmentation,
# and a resumable sampler whose position and RNG state are checkpointed. Spec §6.
#
# Single-process by design (`num_workers=0` default, configurable): Colab's free tier gives only
# 2 vCPUs, so a multi-worker torch DataLoader would spend a meaningful share of those 2 cores on
# IPC/pickling overhead for a dataset that is just numpy slicing (already fast, no PIL/IO-bound
# work per item) — the crossover point where multiprocessing pays for itself needs more cores
# than are available here. `num_workers` is still a constructor parameter so this can be
# revisited if the deploy target changes; anything > 0 is explicitly rejected for now rather than
# silently ignored.
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

MAX_CONCAT_PAIRS = 4  # spec §6: 2-4 pairs joined
CONCAT_STREAM = 1  # third seed word: keeps the concat RNG stream apart from the batch-order RNG


class ShardDataset:
    """In-memory view over every `shard_*.npz` file for one split, matching the PLAN.md shard
    contract: `src`/`tgt` uint16 flat token arrays (no BOS/EOS), `src_off`/`tgt_off` int64
    offsets of length n+1, `ids` a unicode array. `tgt`/`tgt_off` are absent for the `test` split.
    """

    def __init__(
        self, shard_dir: str | Path, split: str, pad_id: int = 0, bos_id: int = 2, eos_id: int = 3
    ):
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        split_dir = Path(shard_dir) / split
        files = sorted(split_dir.glob("shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shard_*.npz files found under {split_dir}")

        self._shards: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        self._ids: list[str] = []
        self._shard_of: list[int] = []
        self._local_idx: list[int] = []
        self.has_tgt = True
        for shard_i, f in enumerate(files):
            with np.load(f) as z:
                src = z["src"]
                src_off = z["src_off"]
                has_tgt = "tgt" in z.files
                tgt = z["tgt"] if has_tgt else None
                tgt_off = z["tgt_off"] if has_tgt else None
                ids = [str(x) for x in z["ids"]]
            if not has_tgt:
                self.has_tgt = False
            self._shards.append((src, src_off, tgt, tgt_off))
            n = len(src_off) - 1
            self._ids.extend(ids)
            self._shard_of.extend([shard_i] * n)
            self._local_idx.extend(range(n))

    def __len__(self) -> int:
        return len(self._ids)

    def src_len(self, i: int) -> int:
        _, src_off, _, _ = self._shards[self._shard_of[i]]
        li = self._local_idx[i]
        return int(src_off[li + 1] - src_off[li])

    def tgt_len(self, i: int) -> int:
        _, _, _, tgt_off = self._shards[self._shard_of[i]]
        if tgt_off is None:
            raise ValueError("this split has no target side (test split)")
        li = self._local_idx[i]
        return int(tgt_off[li + 1] - tgt_off[li])

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray | None, str]:
        src, src_off, tgt, tgt_off = self._shards[self._shard_of[i]]
        li = self._local_idx[i]
        s = src[src_off[li] : src_off[li + 1]]
        t = tgt[tgt_off[li] : tgt_off[li + 1]] if tgt is not None else None
        return s, t, self._ids[i]


@dataclass
class Batch:
    """One materialized, padded training/eval batch. `tgt_in`/`tgt_out` are None for splits
    without a target side (e.g. `test`, used only for translation, not training).
    """

    src: Tensor  # (B, T_src) long, pad-padded, EOS appended per example
    tgt_in: Tensor | None  # (B, T_tgt) long = BOS + tokens, pad-padded
    tgt_out: Tensor | None  # (B, T_tgt) long = tokens + EOS, pad-padded
    ids: list[str]


def batch_token_count(batch: Batch) -> int:
    """Padded token count for grad-accumulation bookkeeping (spec §6: "~25k target tokens per
    optimizer step") and for the max_tokens bucketing budget: batch_size * max(len), counting
    padding as real tokens since that's what actually costs compute/memory.
    """
    n = batch.src.numel()
    if batch.tgt_in is not None:
        n += batch.tgt_in.numel()
    return n


class BucketedSampler:
    """Token-bucketed batching with length-sorted chunks, a shuffled batch order, concatenation
    augmentation, and a resumable (epoch, batch_cursor, rng_state) sampler state.

    Design notes (spec §6 leaves the exact bucketing/resume mechanics unspecified):
    - Per-example length = max(src_len_with_eos, tgt_len_with_bos) (0 if no target side). This
      is the single scalar the spec's "batch_size * max(len)" budget formula refers to; src and
      tgt are padded independently but bucketing on their max keeps both sides' padding waste low
      simultaneously without needing two separate token budgets.
    - Each epoch's batch order (which examples group into which batch, and the shuffled order of
      batches) is a **pure function of (seed, epoch)**: it is rebuilt deterministically by
      `_build_epoch_batches`, never itself persisted. Only `epoch` and `batch_cursor` need to be
      saved to reconstruct "which batch comes next" exactly.
    - Concatenation augmentation is part of the epoch PLAN, decided before bucketing. A
      dedicated RNG keyed on (seed, epoch, CONCAT_STREAM) draws, per example, whether it is
      joined (prob `concat_prob`), how many pairs k in {2,3,4} and the k-1 random partner indices.
      Bucketing then uses the joined length (each side truncated to `concat_max_len`, then the same
      max(src+1, tgt+1) scalar as without concat), so every padded batch obeys the same
      `max_tokens` rule as the no-concat case. (Previously pairs were joined at materialization,
      after bucketing on the unjoined lengths, and `_pad_batch` padded the whole batch to its
      longest joined row: padded micro-batches reached 131k-170k tokens against a 4k-8k budget.)
      With `concat_prob == 0` nothing is drawn and the batches are identical to the pre-plan
      implementation for the same seed.
    - The plan is therefore a pure function of (seed, epoch) like the batch order, so resume needs
      only `epoch` and `batch_cursor`. Old states carrying `aug_rng_state` still load (the key is
      ignored); a pre-plan checkpoint resumed with concat_prob > 0 continues with the new plan
      rather than bit-exactly (only smoke artifacts exist).
    """

    def __init__(
        self,
        dataset: ShardDataset,
        max_tokens: int,
        seed: int,
        concat_prob: float = 0.0,
        concat_max_len: int = 256,
        chunk_size: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        if num_workers != 0:
            raise NotImplementedError(
                "BucketedSampler is single-process by design (see module docstring); "
                "num_workers must be 0"
            )
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.seed = seed
        self.concat_prob = concat_prob
        self.concat_max_len = concat_max_len
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.epoch = 0
        self.batch_cursor = 0
        n = len(dataset)
        self._src_lens = np.array([dataset.src_len(i) for i in range(n)], dtype=np.int64)
        self._tgt_lens = (
            np.array([dataset.tgt_len(i) for i in range(n)], dtype=np.int64)
            if dataset.has_tgt
            else None
        )
        # Per-epoch concat plan (see class docstring): k[i] < 2 means "not joined"; otherwise
        # example i is joined with `partners[i, : k[i] - 1]`.
        self._concat_k = np.zeros(n, dtype=np.int64)
        self._partners = np.zeros((n, MAX_CONCAT_PAIRS - 1), dtype=np.int64)
        self._batches: list[list[int]] = []
        self._build_epoch_batches()

    def _plan_concat(self) -> np.ndarray:
        """Decide this epoch's concatenation plan; return each example's bucketing length
        (max over sides of the joined, truncated length, +1 for EOS/BOS)."""
        n = len(self.dataset)
        self._concat_k = np.zeros(n, dtype=np.int64)
        src_total = self._src_lens.copy()
        tgt_total = self._tgt_lens.copy() if self._tgt_lens is not None else None
        if self.concat_prob > 0.0:
            rng = np.random.default_rng((self.seed, self.epoch, CONCAT_STREAM))
            joined = rng.random(n) < self.concat_prob
            k = rng.integers(2, MAX_CONCAT_PAIRS + 1, size=n)  # 2..4 pairs joined (spec §6)
            self._partners = rng.integers(0, n, size=(n, MAX_CONCAT_PAIRS - 1))
            self._concat_k = np.where(joined, k, 0)
            for col in range(MAX_CONCAT_PAIRS - 1):
                active = self._concat_k > col + 1
                partner = self._partners[:, col]
                src_total += np.where(active, self._src_lens[partner], 0)
                if tgt_total is not None and self._tgt_lens is not None:
                    tgt_total += np.where(active, self._tgt_lens[partner], 0)
            src_total = np.minimum(src_total, self.concat_max_len)
            if tgt_total is not None:
                tgt_total = np.minimum(tgt_total, self.concat_max_len)
        src_len = src_total + 1  # + EOS
        tgt_len = tgt_total + 1 if tgt_total is not None else np.zeros(n, dtype=np.int64)  # + BOS
        return np.maximum(src_len, tgt_len)

    def _build_epoch_batches(self) -> None:
        order_rng = np.random.default_rng((self.seed, self.epoch))
        n = len(self.dataset)
        perm = order_rng.permutation(n) if self.shuffle else np.arange(n)
        ex_len = self._plan_concat()
        batches: list[list[int]] = []
        for start in range(0, n, self.chunk_size):
            chunk = perm[start : start + self.chunk_size]
            lens = ex_len[chunk]
            chunk_sorted = chunk[np.argsort(lens, kind="stable")]
            cur: list[int] = []
            cur_max = 0
            for i in chunk_sorted:
                length = int(ex_len[i])
                new_max = max(cur_max, length)
                if cur and (len(cur) + 1) * new_max > self.max_tokens:
                    batches.append(cur)
                    cur = [int(i)]
                    cur_max = length
                else:
                    cur.append(int(i))
                    cur_max = new_max
            if cur:
                batches.append(cur)
        if self.shuffle and batches:
            batch_order = order_rng.permutation(len(batches))
            batches = [batches[j] for j in batch_order]
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        """Start a new epoch: rebuild its (deterministic) batch order and reset the cursor."""
        self.epoch = epoch
        self.batch_cursor = 0
        self._build_epoch_batches()

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = state["epoch"]
        self.batch_cursor = state["batch_cursor"]
        self._build_epoch_batches()  # batches and concat plan: pure function of (seed, epoch)

    def _materialize(self, indices: list[int]) -> Batch:
        srcs: list[np.ndarray] = []
        tgts: list[np.ndarray] | None = [] if self.dataset.has_tgt else None
        ids: list[str] = []
        for i in indices:
            src, tgt, ex_id = self.dataset.get(i)
            k = int(self._concat_k[i])
            if k >= 2:
                extra = [int(j) for j in self._partners[i, : k - 1]]
                parts = [self.dataset.get(j) for j in extra]
                src = np.concatenate([src] + [p[0] for p in parts])[: self.concat_max_len]
                if tgt is not None:
                    tgt = np.concatenate([tgt] + [p[1] for p in parts])[: self.concat_max_len]
            srcs.append(src)
            if tgts is not None:
                tgts.append(tgt)  # type: ignore[arg-type]
            ids.append(ex_id)
        return self._pad_batch(srcs, tgts, ids)

    def _pad_batch(
        self, srcs: list[np.ndarray], tgts: list[np.ndarray] | None, ids: list[str]
    ) -> Batch:
        pad_id, bos_id, eos_id = self.dataset.pad_id, self.dataset.bos_id, self.dataset.eos_id

        src_seqs = [np.concatenate([s, [eos_id]]) for s in srcs]
        max_src = max(len(s) for s in src_seqs)
        src_tensor = torch.full((len(src_seqs), max_src), pad_id, dtype=torch.long)
        for i, s in enumerate(src_seqs):
            src_tensor[i, : len(s)] = torch.from_numpy(s.astype(np.int64))

        tgt_in_tensor: Tensor | None = None
        tgt_out_tensor: Tensor | None = None
        if tgts is not None:
            tgt_in_seqs = [np.concatenate([[bos_id], t]) for t in tgts]
            tgt_out_seqs = [np.concatenate([t, [eos_id]]) for t in tgts]
            max_tgt = max(len(t) for t in tgt_in_seqs)
            tgt_in_tensor = torch.full((len(tgts), max_tgt), pad_id, dtype=torch.long)
            tgt_out_tensor = torch.full((len(tgts), max_tgt), pad_id, dtype=torch.long)
            for i, (ti, to) in enumerate(zip(tgt_in_seqs, tgt_out_seqs, strict=True)):
                tgt_in_tensor[i, : len(ti)] = torch.from_numpy(ti.astype(np.int64))
                tgt_out_tensor[i, : len(to)] = torch.from_numpy(to.astype(np.int64))

        return Batch(src=src_tensor, tgt_in=tgt_in_tensor, tgt_out=tgt_out_tensor, ids=ids)

    def __iter__(self):
        while self.batch_cursor < len(self._batches):
            batch = self._materialize(self._batches[self.batch_cursor])
            self.batch_cursor += 1
            yield batch
