from __future__ import annotations

"""Training loop: fp16 autocast + GradScaler, AdamW, WSD (warmup-stable-decay)
schedule, label-smoothed cross-entropy, gradient accumulation, resumable
checkpointing (model/optimizer/scaler/scheduler/dataloader/RNG state), checkpoint
averaging and W&B logging. Spec §6.
"""
