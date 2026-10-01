from __future__ import annotations

# Arithmetic check for scripts/epoch_accounting.epochs_block (epochs = steps / steps-per-epoch).
from scripts.epoch_accounting import L4_FALLBACK_STEPS, epochs_block


def test_epochs_block_arithmetic_and_fallback_steps() -> None:
    assert L4_FALLBACK_STEPS == 4107  # floor(2040 / 0.4966)
    replays = [
        {"max_tokens": 4096, "concat_prob": 0.15, "optimizer_steps_per_epoch": 2000},
        {"max_tokens": 8192, "concat_prob": 0.0, "optimizer_steps_per_epoch": 1000},
    ]
    blk = epochs_block(replays)
    assert set(blk) == {"main_l4", "l4_fallback_s3", "3070_s1_s2"}  # no replay -> skipped
    assert blk["main_l4"]["epochs"] == 24645 / 2000
    assert blk["3070_s1_s2"]["arithmetic"] == "2889 / 1000 = 2.889"
