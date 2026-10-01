from __future__ import annotations

# scripts/probe_microbatch.py is CUDA-only by design (spec §14 P5). On CPU-only CI the documented
# no-op fallback is checked: `main()` must not raise and must exit 0. In the CUDA env
# (envs/cuda) the real probe runs on a tiny candidate list and must report a fitting size under
# the allocator cap (sized from free VRAM, as in training), with the optimizer step counted.
import pytest
import torch

from scripts.probe_microbatch import main, probe


@pytest.mark.skipif(torch.cuda.is_available(), reason="no-op fallback only exists without CUDA")
def test_main_is_a_safe_noop_without_cuda() -> None:
    assert main([]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_probe_on_cuda_counts_optimizer_and_respects_cap() -> None:
    result = probe(candidates=(512, 1024), seq_len=64)
    assert result.max_fitting_tokens == 1024
    assert result.includes_optimizer_step
    assert 0.1 <= result.memory_fraction <= 0.95
    assert result.peak_memory_at_max_mb is not None
    assert result.peak_memory_at_max_mb <= result.total_memory_mb * result.memory_fraction
