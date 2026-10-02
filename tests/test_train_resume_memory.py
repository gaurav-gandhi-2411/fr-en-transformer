from __future__ import annotations

# Regression tests for the checkpoint-resume memory bug (RTX 3070, 8 GB: S1 resumed 8 times and
# OOMed at step 6 every time). `load_latest_checkpoint` loaded CUDA-saved checkpoints onto the GPU
# (no map_location) and `train()` kept the whole payload (model + AdamW state) referenced for its
# whole run, i.e. a second full copy of the training state on the device. Fixed by loading to CPU
# and dropping the big payloads right after they are copied into the live objects.
import gc
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch

import nmt.train as nt
from tests.test_train import _make_cfg


def _gpu_idle() -> bool:
    """True only when CUDA exists and no other process holds a compute context (user jobs)."""
    if not torch.cuda.is_available():
        return False
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return not out.strip()


def test_load_latest_checkpoint_loads_to_cpu(tmp_path: Path, monkeypatch) -> None:
    run, ckpt = tmp_path / "run", tmp_path / "ckpt"
    nt.train(_make_cfg("m", run, ckpt), wandb_mode="disabled", synthetic=True, max_steps=10)

    seen: dict[str, Any] = {}
    real_load = torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    payload = nt.load_latest_checkpoint(ckpt)
    assert payload is not None
    assert seen["map_location"] == "cpu"
    assert all(t.device.type == "cpu" for t in payload["model"].values())


def test_resume_does_not_retain_checkpoint_payload(tmp_path: Path, monkeypatch) -> None:
    run, ckpt = tmp_path / "run", tmp_path / "ckpt"
    nt.train(_make_cfg("m", run, ckpt), wandb_mode="disabled", synthetic=True, max_steps=10)

    refs: list[weakref.ref] = []
    real_loader = nt.load_latest_checkpoint

    def spy_loader(ckpt_dir: Path) -> dict[str, Any] | None:
        payload = real_loader(ckpt_dir)
        if payload is not None:
            # Weak refs only: the spy itself must not keep the payload alive.
            refs.extend(weakref.ref(t) for t in payload["model"].values())
        return payload

    monkeypatch.setattr(nt, "load_latest_checkpoint", spy_loader)
    alive_during_training: list[int] = []

    def eval_fn(model: Any, step: int, wandb_run: Any = None) -> dict[str, float]:
        gc.collect()
        alive_during_training.append(sum(r() is not None for r in refs))
        return {}

    cfg = _make_cfg("m", run, ckpt)
    cfg.eval.eval_every = 2
    nt.train(cfg, wandb_mode="disabled", synthetic=True, resume=True, max_steps=14, eval_fn=eval_fn)
    assert refs, "the resume path never loaded a checkpoint"
    assert alive_during_training, "eval_fn never ran after the resume"
    # Before the fix `resumed_ckpt` pinned every model tensor for the entire run.
    assert alive_during_training[0] == 0


@pytest.mark.skipif(not _gpu_idle(), reason="needs CUDA and an idle GPU (no other compute apps)")
def test_cuda_resume_memory_matches_fresh_start(tmp_path: Path) -> None:
    run, ckpt = tmp_path / "run", tmp_path / "ckpt"
    cfg = _make_cfg("m", run, ckpt)
    cfg.device = "cuda"
    cfg.precision = "fp32"
    cfg.eval.eval_every = 0
    peaks: list[int] = []

    def measure(model: Any, step: int, wandb_run: Any = None) -> dict[str, float]:
        gc.collect()
        peaks.append(torch.cuda.memory_allocated())
        return {}

    cfg.eval.eval_every = 12
    torch.cuda.empty_cache()
    nt.train(cfg, wandb_mode="disabled", synthetic=True, max_steps=12, eval_fn=measure)
    fresh = peaks[-1]
    torch.cuda.empty_cache()
    nt.train(cfg, wandb_mode="disabled", synthetic=True, resume=True, max_steps=24, eval_fn=measure)
    resumed = peaks[-1]
    # Same live objects either way; a retained second copy would add ~model+Adam state (~4x params).
    assert resumed <= fresh * 1.10 + 1_000_000
