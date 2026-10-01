from __future__ import annotations

# scripts/probe_microbatch.py is CUDA-only by design (spec §14 P5); this CI/CPU machine has no
# GPU, so the only thing verified here is the documented no-op fallback -- `main()` must not raise
# and must exit 0 when `torch.cuda.is_available()` is False. The real probe logic needs a GPU to
# exercise and is out of scope for a CPU-only gate (consistent with "Do NOT start any Colab/GPU
# run" for this phase).
import torch

from scripts.probe_microbatch import main


def test_main_is_a_safe_noop_without_cuda(capsys: object) -> None:
    assert not torch.cuda.is_available()  # sanity: this test is only meaningful on CPU-only CI
    exit_code = main([])
    assert exit_code == 0
