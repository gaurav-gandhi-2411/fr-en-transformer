from __future__ import annotations

# Tests for nmt/train.py: the WSD schedule's values at key steps, and resume determinism (N
# steps in one run == N/2 + checkpoint + a simulated fresh-process reload + N/2, loss identical
# at every step) built via train.py's own `train()` function on a tiny synthetic shard, dropout
# ON. Spec §12.
import json
import random
from pathlib import Path

import numpy as np
import torch

from nmt.train import (
    BatchSection,
    CkptSection,
    LoggingSection,
    ModelSection,
    OptimSection,
    TrainConfig,
    train,
    wsd_lr_scale,
)


def test_wsd_schedule_key_steps() -> None:
    warmup_steps = 100
    planned_steps = 1000
    cooldown_frac = 0.2  # decay window = 200 steps, starts at step 800

    assert wsd_lr_scale(0, warmup_steps, planned_steps, cooldown_frac) == 0.0
    assert wsd_lr_scale(50, warmup_steps, planned_steps, cooldown_frac) == 0.5  # mid-warmup
    assert wsd_lr_scale(100, warmup_steps, planned_steps, cooldown_frac) == 1.0  # end of warmup
    assert wsd_lr_scale(500, warmup_steps, planned_steps, cooldown_frac) == 1.0  # stable phase
    assert wsd_lr_scale(800, warmup_steps, planned_steps, cooldown_frac) == 1.0  # decay start
    assert wsd_lr_scale(1000, warmup_steps, planned_steps, cooldown_frac) == 0.0  # end

    # Halfway through the 200-step decay window.
    assert abs(wsd_lr_scale(900, warmup_steps, planned_steps, cooldown_frac) - 0.5) < 1e-9


def test_wsd_schedule_cooldown_now_overrides_decay_start() -> None:
    """--cooldown-now: decay starts at an arbitrary earlier step, not planned_steps-derived."""
    scale_at_start = wsd_lr_scale(
        300, warmup_steps=100, planned_steps=1000, cooldown_frac=0.2, decay_start_step=300
    )
    assert scale_at_start == 1.0
    scale_at_end = wsd_lr_scale(
        500, warmup_steps=100, planned_steps=1000, cooldown_frac=0.2, decay_start_step=300
    )
    assert scale_at_end == 0.0


def _make_cfg(name: str, run_dir: Path, ckpt_dir: Path) -> TrainConfig:
    return TrainConfig(
        name=name,
        group="smoke",
        seed=7,
        device="cpu",
        model=ModelSection(
            vocab_size=32, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32, dropout=0.3
        ),
        batch=BatchSection(max_tokens=128, tokens_per_step=128, chunk_size=32),
        optim=OptimSection(lr=1e-3, warmup_steps=2, planned_steps=20, cooldown_frac=0.2),
        ckpt=CkptSection(dir=str(ckpt_dir), ckpt_minutes=999, ckpt_steps=10),
        logging=LoggingSection(run_dir=str(run_dir)),
    )


def _read_step_losses(run_dir: Path) -> dict[int, float]:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in rows if "eval" not in r}


def test_resume_determinism(tmp_path: Path) -> None:
    """A full 20-step run must match a 10+10 split run (checkpoint at step 10, RNG scrambled to
    simulate a fresh process, then resumed) at every step, within 1e-4 — with dropout ON, so this
    genuinely exercises the saved torch RNG state, not just a deterministic-by-construction path.
    """
    run_a, ckpt_a = tmp_path / "a", tmp_path / "a_ckpt"
    train(_make_cfg("a", run_a, ckpt_a), wandb_mode="disabled", synthetic=True, max_steps=20)

    run_b, ckpt_b = tmp_path / "b", tmp_path / "b_ckpt"
    train(_make_cfg("b", run_b, ckpt_b), wandb_mode="disabled", synthetic=True, max_steps=10)

    # Simulate a fresh process: scramble every RNG train() itself doesn't own, so a pass here can
    # only be explained by the checkpoint's saved RNG state actually being restored.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    train(
        _make_cfg("b", run_b, ckpt_b),
        wandb_mode="disabled",
        synthetic=True,
        resume=True,
        max_steps=20,
    )

    losses_a = _read_step_losses(run_a)
    losses_b = _read_step_losses(run_b)
    assert set(losses_a) == set(losses_b) == set(range(1, 21))
    for step in losses_a:
        diff = abs(losses_a[step] - losses_b[step])
        assert diff < 1e-4, f"step {step}: {losses_a[step]} vs {losses_b[step]}"


def test_resume_without_existing_checkpoint_starts_fresh(tmp_path: Path) -> None:
    """--resume with an empty ckpt_dir must not error; it just starts a fresh run."""
    run_dir, ckpt_dir = tmp_path / "run", tmp_path / "ckpt"
    train(
        _make_cfg("fresh", run_dir, ckpt_dir),
        wandb_mode="disabled",
        synthetic=True,
        resume=True,
        max_steps=3,
    )
    losses = _read_step_losses(run_dir)
    assert set(losses) == {1, 2, 3}


def test_resume_determinism_with_concat_augmentation(tmp_path: Path) -> None:
    """Same 20 == 10+10 loss-trajectory check with concat augmentation on: the concat plan is a
    pure function of (seed, epoch), so resume reproduces it with no saved augmentation RNG."""

    def cfg(name: str, run: Path, ckpt: Path) -> TrainConfig:
        c = _make_cfg(name, run, ckpt)
        c.batch.concat_prob = 0.5
        c.batch.concat_max_len = 24
        return c

    run_a, ckpt_a = tmp_path / "a", tmp_path / "a_ckpt"
    train(cfg("a", run_a, ckpt_a), wandb_mode="disabled", synthetic=True, max_steps=20)
    run_b, ckpt_b = tmp_path / "b", tmp_path / "b_ckpt"
    train(cfg("b", run_b, ckpt_b), wandb_mode="disabled", synthetic=True, max_steps=10)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    train(cfg("b", run_b, ckpt_b), wandb_mode="disabled", synthetic=True, resume=True, max_steps=20)
    losses_a, losses_b = _read_step_losses(run_a), _read_step_losses(run_b)
    assert set(losses_a) == set(losses_b) == set(range(1, 21))
    for step in losses_a:
        assert abs(losses_a[step] - losses_b[step]) < 1e-4, f"step {step}"
