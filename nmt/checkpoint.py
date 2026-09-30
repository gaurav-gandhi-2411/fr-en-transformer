from __future__ import annotations

# Checkpoint averaging: average model weights across the last N checkpoints (Vaswani et al.
# 2017), typically the decay-phase checkpoints saved during the WSD schedule's cooldown. Spec §6,
# §9. Kept separate from nmt/train.py so `nmt.evaluate`/`nmt.hub` (P4/export) can average and
# load a champion checkpoint without importing the training loop.
from pathlib import Path

import torch


def average_checkpoints(ckpt_paths: list[Path]) -> dict[str, torch.Tensor]:
    """Return a state_dict that is the element-wise mean of the `model` state_dicts stored in
    each checkpoint file (as written by `nmt.train.save_checkpoint`). All checkpoints must share
    the same model architecture (identical parameter names/shapes).
    """
    if not ckpt_paths:
        raise ValueError("average_checkpoints requires at least one checkpoint path")
    n = len(ckpt_paths)
    avg: dict[str, torch.Tensor] = {}
    for path in ckpt_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model"]
        for key, value in state.items():
            value = value.float() / n
            avg[key] = value if key not in avg else avg[key] + value
    return avg
