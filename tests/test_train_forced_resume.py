from __future__ import annotations

# Forced-resume contract of `python -m nmt.train` (the Colab notebook greps these exact lines):
# --stop-after-first-ckpt, --ckpt-steps, CKPT_SAVED / RESUMED / RESUME_CONTEXT / POST_RESUME and
# the periodic step line. A fresh run stops at its first checkpoint, the second invocation resumes
# with bit-identical RNG/sampler state (matches_saved=True) and the combined loss trajectory
# equals an uninterrupted run (same pattern as tests/test_train.py::test_resume_determinism).
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nmt.train import main as train_main


def _write_config(path: Path, run_dir: Path, ckpt_dir: Path) -> None:
    cfg = {
        "name": "forced_resume",
        "group": "smoke",
        "seed": 7,
        "device": "cpu",
        "model": {
            "vocab_size": 32,
            "d_model": 16,
            "n_heads": 2,
            "enc_layers": 1,
            "dec_layers": 1,
            "d_ff": 32,
            "dropout": 0.3,  # dropout ON so the saved torch RNG state genuinely matters
        },
        "batch": {"max_tokens": 128, "tokens_per_step": 128, "chunk_size": 32},
        "optim": {"lr": 1e-3, "warmup_steps": 2, "planned_steps": 20, "cooldown_frac": 0.2},
        "ckpt": {"dir": str(ckpt_dir), "ckpt_minutes": 999, "ckpt_steps": None},
        "logging": {"run_dir": str(run_dir), "log_every": 4},
        "eval": {"eval_every": 100000},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _losses(run_dir: Path) -> dict[int, float]:
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in rows if "loss" in r and "eval" not in r}


def _run(cfg_path: Path, *extra: str) -> None:
    train_main(["--config", str(cfg_path), "--synthetic", "--wandb", "disabled", *extra])


def test_stop_after_first_ckpt_then_resume_matches_uninterrupted_run(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run_a, ckpt_a = tmp_path / "a", tmp_path / "a_ckpt"
    cfg_a = tmp_path / "a.yaml"
    _write_config(cfg_a, run_a, ckpt_a)
    _run(cfg_a, "--max-steps", "12")  # uninterrupted reference
    capsys.readouterr()

    run_b, ckpt_b = tmp_path / "b", tmp_path / "b_ckpt"
    cfg_b = tmp_path / "b.yaml"
    _write_config(cfg_b, run_b, ckpt_b)

    # --- first invocation: fresh, stops cleanly right after the first checkpoint (step 5) -----
    _run(cfg_b, "--max-steps", "12", "--ckpt-steps", "5", "--stop-after-first-ckpt")
    out1 = capsys.readouterr().out
    saved = re.search(r"CKPT_SAVED step=5 rng_fingerprint=([0-9a-f]{12}) path=(\S+)", out1)
    assert saved, out1
    assert f"STOPPED_AFTER_FIRST_CKPT step=5 path={saved.group(2)}" in out1
    assert Path(saved.group(2)).is_file()
    assert sorted(_losses(run_b)) == [1, 2, 3, 4, 5]  # stopped: nothing past the checkpoint
    assert "step 4 loss=" in out1  # periodic stdout progress line (log_every=4)
    assert "tok/s=" in out1 and "peak_mem=" in out1 and "step_time=" in out1

    # --- second invocation: resumes, flag ignored, continues to the end ---------------------
    random.seed(999)  # fresh-process stand-in: scramble every RNG train() doesn't own
    np.random.seed(999)
    torch.manual_seed(999)
    _run(cfg_b, "--max-steps", "12", "--ckpt-steps", "5", "--stop-after-first-ckpt", "--resume")
    out2 = capsys.readouterr().out
    assert "stop-after-first-ckpt ignored: resumed run" in out2
    assert "STOPPED_AFTER_FIRST_CKPT" not in out2
    resumed = re.search(r"RESUMED step=5 rng_fingerprint=([0-9a-f]{12}) matches_saved=(\w+)", out2)
    assert resumed, out2
    assert resumed.group(2) == "True"
    assert resumed.group(1) == saved.group(1)  # live state after restore == state at save time

    losses_b = _losses(run_b)
    context = re.search(r"RESUME_CONTEXT last_losses=(\[.*?\])", out2)
    assert context and json.loads(context.group(1)) == pytest.approx(
        [losses_b[s] for s in (3, 4, 5)], abs=1e-5
    )
    post = re.search(r"POST_RESUME losses=(\[.*?\])", out2)
    assert post and json.loads(post.group(1)) == pytest.approx(
        [losses_b[s] for s in (6, 7, 8)], abs=1e-5
    )

    # --- the stitched trajectory equals the uninterrupted one -------------------------------
    losses_a = _losses(run_a)
    assert set(losses_a) == set(losses_b) == set(range(1, 13))
    for step in losses_a:
        assert abs(losses_a[step] - losses_b[step]) < 1e-4, step


def test_resume_with_no_checkpoint_is_fresh_so_stop_flag_applies(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """--resume with an empty ckpt dir is a FRESH start, so the stop flag still applies."""
    cfg = tmp_path / "c.yaml"
    _write_config(cfg, tmp_path / "run", tmp_path / "ckpt")
    _run(cfg, "--max-steps", "9", "--ckpt-steps", "3", "--stop-after-first-ckpt", "--resume")
    out = capsys.readouterr().out
    assert "STOPPED_AFTER_FIRST_CKPT step=3" in out
    assert "RESUMED step=" not in out and "stop-after-first-ckpt ignored" not in out
