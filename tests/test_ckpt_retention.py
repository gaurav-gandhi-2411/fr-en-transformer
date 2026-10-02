from __future__ import annotations

# Regression for the main-run retention bug (PREREG 2026-10-02 (b) disclosure): `keep_decay_phase`
# only protected decay-phase checkpoints when `decay_start_step` was set, which happened only
# under --cooldown-now, so the main run kept just its last `keep_last` files.
from pathlib import Path

import pytest

from nmt.train import (
    BatchSection,
    CkptSection,
    LoggingSection,
    ModelSection,
    OptimSection,
    TrainConfig,
    prune_checkpoints,
    train,
    wsd_decay_window,
)

# The main run's real numbers: planned 24,645, cooldown 0.2, warmup 4,000 (decay starts 19,716).
MAIN_PLANNED, MAIN_COOLDOWN, MAIN_WARMUP = 24645, 0.2, 4000


def _steps(ckpt_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("step_*.pt"))


@pytest.mark.parametrize("every", [1700, 500])
def test_main_run_numbers_decay_phase_checkpoints_survive_pruning(
    tmp_path: Path, every: int
) -> None:
    """Replays save -> prune with the main run's schedule. At every=1700 only 4 decay-phase files
    exist (<= keep_last=5), so the old code lost nothing; at every=500 there are 10 and the old
    code kept only 5 of them, which is the case `keep_decay_phase` exists for."""
    decay_start, decay_len = wsd_decay_window(MAIN_WARMUP, MAIN_PLANNED, MAIN_COOLDOWN)
    assert (decay_start, decay_len) == (19716, 4929)
    saved = [*range(every, MAIN_PLANNED, every), MAIN_PLANNED]
    for step in saved:
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
        prune_checkpoints(tmp_path, 5, decay_start, True)  # the call train() makes after a save
    survivors = _steps(tmp_path)
    decay_phase = [s for s in saved if s >= decay_start]
    assert set(decay_phase) <= set(survivors)
    assert set(saved[-5:]) <= set(survivors)  # keep_last still holds
    assert set(survivors) == set(decay_phase) | set(saved[-5:])  # nothing else is kept
    if every == 1700:
        assert decay_phase == [20400, 22100, 23800, 24645]


def test_keep_decay_phase_false_still_keeps_only_last_n(tmp_path: Path) -> None:
    for step in range(1000, 9000, 1000):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    prune_checkpoints(tmp_path, 3, 5000, False)
    assert _steps(tmp_path) == [6000, 7000, 8000]


def _cfg(run_dir: Path, ckpt_dir: Path, keep_decay_phase: bool) -> TrainConfig:
    return TrainConfig(
        name="retention",
        group="smoke",
        seed=7,
        device="cpu",
        model=ModelSection(
            vocab_size=32, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32, dropout=0.1
        ),
        batch=BatchSection(max_tokens=128, tokens_per_step=128, chunk_size=32),
        # decay window = round(40 * 0.25) = 10 steps, starting at 30 (no --cooldown-now).
        optim=OptimSection(lr=1e-3, warmup_steps=2, planned_steps=40, cooldown_frac=0.25),
        ckpt=CkptSection(
            dir=str(ckpt_dir),
            ckpt_minutes=999,
            ckpt_steps=4,
            keep_last=2,
            keep_decay_phase=keep_decay_phase,
        ),
        logging=LoggingSection(run_dir=str(run_dir)),
    )


def test_train_retains_decay_phase_checkpoints_without_cooldown_now(tmp_path: Path) -> None:
    train(_cfg(tmp_path / "r", tmp_path / "c", True), wandb_mode="disabled", synthetic=True)
    # saved at 4, 8, ..., 40; decay starts at 30 -> 32, 36, 40 are protected; last 2 = 36, 40.
    assert _steps(tmp_path / "c") == [32, 36, 40]


def test_train_prunes_to_last_n_when_keep_decay_phase_is_false(tmp_path: Path) -> None:
    train(_cfg(tmp_path / "r", tmp_path / "c", False), wandb_mode="disabled", synthetic=True)
    assert _steps(tmp_path / "c") == [36, 40]
