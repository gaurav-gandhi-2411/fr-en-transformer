from __future__ import annotations

# Training loop: fp16 autocast + GradScaler, AdamW, WSD (warmup-stable-decay)
# schedule, label-smoothed cross-entropy, gradient accumulation, resumable
# checkpointing (model/optimizer/scaler/scheduler/dataloader/RNG state), checkpoint
# averaging and W&B logging. Spec §6.
#
# CLI: `python -m nmt.train --config configs/X.yaml [--resume] [--seed 1234] [--max-steps N]
# [--wandb offline|online|disabled] [--cooldown-now] [--synthetic]`.
#
# Config is plain dataclasses, not pydantic: pydantic is not a pinned dependency in this repo
# (pyproject.toml/uv.lock) and the standing rule is "no new dependencies without asking" — adding
# one mid-task would stall P3 on an approval round-trip for a YAML shape that plain dataclasses
# validate perfectly well by hand. Flagged in the P3 report as a deviation from the general
# pydantic-for-config-boundaries preference.
import argparse
import contextlib
import dataclasses
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from nmt.data.loader import Batch, BucketedSampler, ShardDataset, batch_token_count
from nmt.data.synthetic import write_synthetic_shard_dir
from nmt.model.transformer import ModelConfig, Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------


@dataclass
class ModelSection:
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 8
    dec_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    pos: str = "rope"
    max_len: int = 512
    rope_base: float = 10000.0


@dataclass
class DataSection:
    shard_dir: str = "data/shards"
    train_split: str = "train"
    eval_split: str = "e1"


@dataclass
class BatchSection:
    max_tokens: int = 4096
    tokens_per_step: int = 25000  # target tokens per optimizer step (grad accumulation, spec §6)
    concat_prob: float = 0.0
    concat_max_len: int = 256
    chunk_size: int = 512
    num_workers: int = 0


@dataclass
class OptimSection:
    lr: float = 7.0e-4
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1.0e-9
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 4000
    planned_steps: int = 300  # TBD from pilot for pilot/ablation/main configs; see configs/*.yaml
    cooldown_frac: float = 0.2  # final fraction of planned_steps spent decaying to 0
    max_minutes: float | None = None  # wall-clock cap (pilot=15, ablations=40, main safety=240)


@dataclass
class CkptSection:
    dir: str = "runs/main/ckpt"
    ckpt_minutes: float = 15.0
    ckpt_steps: int | None = None
    keep_last: int = 5
    keep_decay_phase: bool = True


@dataclass
class LoggingSection:
    wandb_mode: str = "offline"
    wandb_project: str = "fr-en-transformer"
    log_every: int = 50
    run_dir: str = "runs/main"


@dataclass
class EvalSection:
    eval_every: int = 500


@dataclass
class TrainConfig:
    name: str
    group: str = "main"  # pilot | ablation | main | smoke (spec §11 W&B run groups)
    seed: int = 1234
    label_smoothing: float = 0.1
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    batch: BatchSection = field(default_factory=BatchSection)
    optim: OptimSection = field(default_factory=OptimSection)
    ckpt: CkptSection = field(default_factory=CkptSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    eval: EvalSection = field(default_factory=EvalSection)


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a YAML training config into a `TrainConfig`. Unknown top-level or
    section keys raise (a mistyped key would otherwise silently fall back to a default).
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section_types: dict[str, type] = {
        "model": ModelSection,
        "data": DataSection,
        "batch": BatchSection,
        "optim": OptimSection,
        "ckpt": CkptSection,
        "logging": LoggingSection,
        "eval": EvalSection,
    }
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in section_types:
            section_cls = section_types[key]
            valid_fields = {f.name for f in dataclasses.fields(section_cls)}
            unknown = set(value or {}) - valid_fields
            if unknown:
                raise ValueError(f"unknown key(s) in config section {key!r}: {sorted(unknown)}")
            kwargs[key] = section_cls(**(value or {}))
        else:
            kwargs[key] = value
    valid_top = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown_top = set(kwargs) - valid_top
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")
    return TrainConfig(**kwargs)


# ---------------------------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA). `warn_only=True` because a couple of ops used
    here (e.g. embedding backward with a padding_idx) don't have a deterministic CUDA kernel;
    warn rather than hard-fail so the same code path runs on both CPU (fully deterministic) and
    a future GPU run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_sha() -> str | None:
    """Best-effort git SHA for run provenance; tolerates failure (spec §11)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# WSD schedule
# ---------------------------------------------------------------------------------------------


def wsd_lr_scale(
    step: int,
    warmup_steps: int,
    planned_steps: int,
    cooldown_frac: float,
    decay_start_step: int | None = None,
) -> float:
    """Warmup-stable-decay LR multiplier in [0, 1] (Hägele et al. 2024), `step` 0-indexed.

    Linear warmup to 1.0 over `warmup_steps`, stable at 1.0, then linear decay to 0 over
    `cooldown_frac * planned_steps` steps. By default the decay window ends exactly at
    `planned_steps`; `--cooldown-now` instead passes an explicit `decay_start_step` (the step at
    which cooldown was triggered), letting a run stop and cool down from any checkpoint (spec §6:
    "robust to Colab cutoffs").
    """
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    decay_len = max(1, round(planned_steps * cooldown_frac))
    if decay_start_step is None:
        decay_start_step = max(warmup_steps, planned_steps - decay_len)
    if step < decay_start_step:
        return 1.0
    if step >= decay_start_step + decay_len:
        return 0.0
    return 1.0 - (step - decay_start_step) / decay_len


# ---------------------------------------------------------------------------------------------
# Loss and optimizer
# ---------------------------------------------------------------------------------------------


def label_smoothed_nll_loss(logits: Tensor, target: Tensor, pad_id: int, epsilon: float) -> Tensor:
    """Label-smoothed cross-entropy (Vaswani et al. 2017 §5.4 / Szegedy et al. 2016), ignoring
    pad positions. `epsilon=0` reduces to plain mean CE over non-pad tokens (used by the
    tiny-model-overfits-one-batch test, where a smoothed loss can't reach 0.1 by construction
    since it floors above 0 even at a perfect one-hot prediction).
    """
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    per_token = (1.0 - epsilon) * nll + epsilon * smooth
    mask = (target != pad_id).to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW weight decay excludes norms and biases (spec §6), not embeddings — embedding
    weight is 2-D so it lands in the decay group same as any other Linear weight.
    """
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() == 1 else decay).append(p)  # 1-D == LayerNorm weight/bias or a bias
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------------------------


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sampler: BucketedSampler,
    decay_start_step: int | None,
    grad_skip_count: int,
) -> Path:
    """Atomic checkpoint write (tmp file + os.replace), so a crash mid-write never leaves a
    corrupt file that `load_latest_checkpoint` could pick up. Includes model, optimizer, scaler,
    sampler state and ALL RNG states (python/numpy/torch CPU/torch CUDA), per spec §6.
    """
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "sampler": sampler.state_dict(),
        "decay_start_step": decay_start_step,
        "grad_skip_count": grad_skip_count,
        "rng": _rng_state(),
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / f"step_{step:08d}.pt"
    tmp_path = ckpt_dir / f".tmp_step_{step:08d}.pt"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def load_latest_checkpoint(ckpt_dir: Path) -> dict[str, Any] | None:
    files = sorted(Path(ckpt_dir).glob("step_*.pt"))
    if not files:
        return None
    # weights_only=False: checkpoints carry optimizer/sampler/RNG state, not just tensors — these
    # are our own trusted, locally-written files, never untrusted third-party checkpoints.
    return torch.load(files[-1], weights_only=False)


def prune_checkpoints(
    ckpt_dir: Path, keep_last: int, decay_start_step: int | None, keep_decay_phase: bool
) -> None:
    """Delete old checkpoints beyond `keep_last`, but never delete ones at/after
    `decay_start_step` when `keep_decay_phase` is set — checkpoint averaging (spec §6, §9) needs
    the decay-phase checkpoints to still be on disk regardless of how many steps have passed.
    """
    files = sorted(Path(ckpt_dir).glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if len(files) <= keep_last:
        return
    protected = {
        f
        for f in files
        if keep_decay_phase
        and decay_start_step is not None
        and int(f.stem.split("_")[1]) >= decay_start_step
    }
    candidates = [f for f in files if f not in protected]
    n_to_delete = max(0, len(files) - keep_last)
    for f in candidates[:n_to_delete]:
        f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------------------------


def _wandb_authenticated() -> bool:
    """Best-effort, network-free check for an available W&B API key (env var or netrc)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = Path(os.environ.get("NETRC", Path.home() / ".netrc"))
    if netrc_path.is_file():
        try:
            return "api.wandb.ai" in netrc_path.read_text(encoding="utf-8")
        except OSError:
            return False
    return False


def _resolve_wandb_mode(requested: str) -> str:
    if requested == "online" and not _wandb_authenticated():
        print(
            "wandb: no API key found (env WANDB_API_KEY / ~/.netrc); falling back to offline mode."
        )
        return "offline"
    return requested


def init_wandb(
    cfg: TrainConfig, mode: str, seed: int, model: Transformer, device: torch.device
) -> Any:
    """Initialize a W&B run in the requested mode (offline/online/disabled), or return None when
    disabled or when the `wandb` package/init fails for any reason (never blocks training).
    """
    mode = _resolve_wandb_mode(mode)
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb: package not importable; continuing without W&B logging.")
        return None

    wandb_dir = REPO_ROOT / "wandb"
    wandb_dir.mkdir(exist_ok=True)
    try:
        return wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.name,
            group=cfg.group,
            mode=mode,
            dir=str(wandb_dir),
            config={
                "seed": seed,
                "git_sha": _git_sha(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "param_count": model.param_count(),
                **{k: v for k, v in dataclasses.asdict(cfg).items()},
            },
        )
    except Exception as exc:  # noqa: BLE001 - W&B failures must never abort training
        print(f"wandb: init failed ({exc!r}); continuing without W&B logging.")
        return None


# ---------------------------------------------------------------------------------------------
# Eval hook (P4 fills this in)
# ---------------------------------------------------------------------------------------------

EvalFn = Callable[[nn.Module, int], dict[str, float]]


def default_eval_fn(model: nn.Module, step: int) -> dict[str, float]:
    """No-op eval hook placeholder. TODO(P4 owner): replace with `nmt.evaluate`'s greedy
    BLEU/chrF computation on E1/E2/dev (spec §8, §11). Validation loss on E1 is computed directly
    in the train loop below (`_eval_val_loss`), not through this hook, since it's needed for
    checkpoint-selection sanity even before P4 exists.
    """
    del model, step
    return {}


@torch.no_grad()
def _eval_val_loss(
    model: Transformer,
    eval_dataset: ShardDataset,
    cfg: TrainConfig,
    device: torch.device,
    max_batches: int = 20,
) -> float | None:
    """Mean label-smoothed loss over (up to) `max_batches` batches of the eval split, or None if
    the eval split has no target side to score against.
    """
    if not eval_dataset.has_tgt:
        return None
    was_training = model.training
    model.eval()
    sampler = BucketedSampler(
        eval_dataset, max_tokens=cfg.batch.max_tokens, seed=0, shuffle=False, concat_prob=0.0
    )
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(sampler):
        if i >= max_batches:
            break
        assert batch.tgt_in is not None and batch.tgt_out is not None
        src = batch.src.to(device)
        tgt_in = batch.tgt_in.to(device)
        tgt_out = batch.tgt_out.to(device)
        logits = model(src, tgt_in)
        # pad_id=0 is the fixed contract (PLAN.md interface contracts), not config-driven.
        loss = label_smoothed_nll_loss(logits, tgt_out, 0, cfg.label_smoothing)
        ntokens = int((tgt_out != 0).sum().item())
        total_loss += loss.item() * ntokens
        total_tokens += ntokens
    if was_training:
        model.train()
    return total_loss / total_tokens if total_tokens else None


# ---------------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------------


def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model_config(cfg: TrainConfig) -> ModelConfig:
    return ModelConfig(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        enc_layers=cfg.model.enc_layers,
        dec_layers=cfg.model.dec_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        pos=cfg.model.pos,
        max_len=cfg.model.max_len,
        rope_base=cfg.model.rope_base,
    )


def _iter_micro_batches(sampler: BucketedSampler) -> Iterator[Batch]:
    """Infinite iterator over `sampler`'s batches, advancing `sampler.epoch` automatically when
    one epoch's batches are exhausted (epoch batches are a pure function of (seed, epoch), so
    this never needs to be resumed itself — the sampler's own `epoch`/`batch_cursor` in the
    checkpoint is enough to reconstruct where in this infinite sequence training resumes).
    """
    while True:
        yield from sampler
        sampler.set_epoch(sampler.epoch + 1)


def _resolve_shard_dir(cfg: TrainConfig, synthetic: bool, run_dir: Path, seed: int) -> Path:
    if not synthetic:
        return Path(cfg.data.shard_dir)
    synth_dir = run_dir / "synthetic_shards"
    write_synthetic_shard_dir(
        synth_dir, vocab_size=cfg.model.vocab_size, seed=seed, n_train=512, n_eval=64
    )
    return synth_dir


def train(
    cfg: TrainConfig,
    *,
    resume: bool = False,
    max_steps: int | None = None,
    wandb_mode: str = "offline",
    cooldown_now: bool = False,
    synthetic: bool = False,
    seed: int | None = None,
    eval_fn: EvalFn = default_eval_fn,
    print_at_steps: tuple[int, ...] = (),
) -> None:
    """Run (or resume) training for `cfg`. Every call builds a brand-new model/optimizer/sampler
    from scratch and, if `resume=True`, restores them from the latest checkpoint in
    `cfg.ckpt.dir` — this mirrors a real process restart exactly (no in-memory state survives
    between calls), which is what makes the resume-determinism test meaningful.
    """
    seed = cfg.seed if seed is None else seed
    device = _resolve_device(cfg)
    seed_everything(seed)

    run_dir = Path(cfg.logging.run_dir)
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.ckpt.dir)
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else REPO_ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    shard_dir = _resolve_shard_dir(cfg, synthetic, run_dir, seed)
    train_dataset = ShardDataset(shard_dir, cfg.data.train_split)
    model = Transformer(_build_model_config(cfg)).to(device)
    sampler = BucketedSampler(
        train_dataset,
        max_tokens=cfg.batch.max_tokens,
        seed=seed,
        concat_prob=cfg.batch.concat_prob,
        concat_max_len=cfg.batch.concat_max_len,
        chunk_size=cfg.batch.chunk_size,
        num_workers=cfg.batch.num_workers,
    )
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.optim.weight_decay),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    step = 0
    decay_start_step: int | None = None
    grad_skip_count = 0
    if resume:
        ckpt = load_latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            sampler.load_state_dict(ckpt["sampler"])
            _restore_rng_state(ckpt["rng"])
            step = ckpt["step"]
            decay_start_step = ckpt.get("decay_start_step")
            grad_skip_count = ckpt.get("grad_skip_count", 0)
    if cooldown_now and decay_start_step is None:
        decay_start_step = step

    wandb_run = init_wandb(cfg, wandb_mode, seed, model, device)

    target_steps = max_steps if max_steps is not None else cfg.optim.planned_steps
    eval_dataset: ShardDataset | None = None
    # eval split not present yet (e.g. real P2 shards not built); skip eval-loss logging.
    with contextlib.suppress(FileNotFoundError):
        eval_dataset = ShardDataset(shard_dir, cfg.data.eval_split)

    micro_iter = _iter_micro_batches(sampler)
    last_ckpt_time = time.monotonic()
    start_time = time.monotonic()
    model.train()

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        while step < target_steps:
            elapsed_minutes = (time.monotonic() - start_time) / 60.0
            if cfg.optim.max_minutes is not None and elapsed_minutes >= cfg.optim.max_minutes:
                break

            micro_batches: list[Batch] = []
            tokens = 0
            while tokens < cfg.batch.tokens_per_step:
                micro_batches.append(next(micro_iter))
                tokens += batch_token_count(micro_batches[-1])

            step += 1
            lr_scale = wsd_lr_scale(
                step - 1,
                cfg.optim.warmup_steps,
                cfg.optim.planned_steps,
                cfg.optim.cooldown_frac,
                decay_start_step,
            )
            lr = cfg.optim.lr * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = lr

            total_ntokens = sum(
                int((b.tgt_out != 0).sum().item()) for b in micro_batches if b.tgt_out is not None
            )
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_start = time.monotonic()
            for b in micro_batches:
                assert b.tgt_in is not None and b.tgt_out is not None
                src, tgt_in, tgt_out = b.src.to(device), b.tgt_in.to(device), b.tgt_out.to(device)
                autocast_on = device.type == "cuda"  # fp16 autocast only on GPU; CPU stays fp32
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=autocast_on
                ):
                    logits = model(src, tgt_in)
                    loss = label_smoothed_nll_loss(logits, tgt_out, 0, cfg.label_smoothing)
                ntokens = int((tgt_out != 0).sum().item())
                weight = ntokens / max(1, total_ntokens)
                scaler.scale(loss * weight).backward()
                step_loss += loss.item() * weight

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            stepped = bool(torch.isfinite(grad_norm))
            if stepped:
                scaler.step(optimizer)
            else:
                grad_skip_count += 1
            scaler.update()

            tok_per_sec = tokens / max(1e-6, time.monotonic() - step_start)
            gpu_mem_mb = (
                torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
            )
            epoch_fraction = sampler.epoch + (sampler.batch_cursor / max(1, len(sampler)))
            row = {
                "step": step,
                "loss": step_loss,
                "ppl": float(min(1e9, np.exp(step_loss))),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "loss_scale": float(scaler.get_scale()),
                "tok_per_sec": tok_per_sec,
                "gpu_mem_mb": gpu_mem_mb,
                "epoch_fraction": epoch_fraction,
                "grad_skip_count": grad_skip_count,
                "optimizer_stepped": stepped,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

            if step in print_at_steps:
                print(f"step {step}: loss={step_loss:.4f}")

            if step % cfg.logging.log_every == 0 and wandb_run is not None:
                wandb_run.log(row, step=step)

            eval_due = eval_dataset is not None and cfg.eval.eval_every > 0
            if eval_due and step % cfg.eval.eval_every == 0:
                val_loss = _eval_val_loss(model, eval_dataset, cfg, device)  # type: ignore[arg-type]
                extra = eval_fn(model, step)
                eval_row = {"step": step, "val_loss": val_loss, **extra}
                metrics_file.write(json.dumps({"eval": eval_row}) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    log_row = {f"eval/{k}": v for k, v in eval_row.items() if k != "step"}
                    wandb_run.log(log_row, step=step)

            due_by_time = (time.monotonic() - last_ckpt_time) / 60.0 >= cfg.ckpt.ckpt_minutes
            due_by_steps = cfg.ckpt.ckpt_steps is not None and step % cfg.ckpt.ckpt_steps == 0
            if due_by_time or due_by_steps or step >= target_steps:
                save_checkpoint(
                    ckpt_dir,
                    step,
                    model,
                    optimizer,
                    scaler,
                    sampler,
                    decay_start_step,
                    grad_skip_count,
                )
                prune_checkpoints(
                    ckpt_dir, cfg.ckpt.keep_last, decay_start_step, cfg.ckpt.keep_decay_phase
                )
                last_ckpt_time = time.monotonic()

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FR->EN transformer (spec §6).")
    parser.add_argument("--config", required=True, help="Path to a YAML config (configs/*.yaml).")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint in ckpt.dir."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override optim.planned_steps.")
    parser.add_argument("--wandb", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument(
        "--cooldown-now", action="store_true", help="Start WSD decay at the current step."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Train on a generated copy-task instead of real shards.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    print_steps = (1, 50, 100, 150, 200, 250, 300) if cfg.name == "smoke" else ()
    train(
        cfg,
        resume=args.resume,
        max_steps=args.max_steps,
        wandb_mode=args.wandb,
        cooldown_now=args.cooldown_now,
        synthetic=args.synthetic,
        seed=args.seed,
        print_at_steps=print_steps,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
