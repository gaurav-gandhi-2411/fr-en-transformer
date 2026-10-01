from __future__ import annotations

# Training loop: autocast (bf16 / fp16 + GradScaler / fp32, config key `precision`), AdamW, WSD
# (warmup-stable-decay) schedule, label-smoothed cross-entropy, gradient accumulation, resumable
# checkpointing (model/optimizer/scaler/scheduler/dataloader/RNG state), checkpoint
# averaging and W&B logging. Spec §6.
#
# CLI: `python -m nmt.train --config configs/X.yaml [--resume] [--seed 1234] [--max-steps N]
# [--wandb offline|online|disabled] [--cooldown-now] [--synthetic] [--precision P] [--device D]
# [--ckpt-steps N] [--stop-after-first-ckpt]`.
#
# Precision (spec §6 said fp16-only because the Colab T4 has no bf16): `auto` picks bf16 on CUDA
# when the GPU supports it (no GradScaler needed), else fp16 + GradScaler, and fp32 on CPU. TF32
# is enabled for CUDA fp32 matmuls/convs. torch.compile is deliberately NOT used anywhere
# (`TORCH_COMPILE = False`; enforced by tests/test_train_precision.py): the Windows/3070 setup has
# no Triton, and a compile graph per sequence-length bucket would dominate a 15-40 min run.
#
# Determinism: `torch.use_deterministic_algorithms(True, warn_only=True)` plus, on CUDA,
# CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before the first cuBLAS call). That fixed cuBLAS workspace
# removes cuBLAS's nondeterministic stream-dependent algorithm choices at some throughput cost
# (the size of the cost is NOT measured here; it is always on in training runs, so the pilot's
# tok/s already includes it). Ops lacking a deterministic CUDA kernel only warn, and that warning
# is deduplicated to once per process.
#
# Config is plain dataclasses, not pydantic: pydantic is not a pinned dependency in this repo
# (pyproject.toml/uv.lock) and the standing rule is "no new dependencies without asking" — adding
# one mid-task would stall P3 on an approval round-trip for a YAML shape that plain dataclasses
# validate perfectly well by hand. Flagged in the P3 report as a deviation from the general
# pydantic-for-config-boundaries preference.
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import warnings
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

# torch.compile stays off (see module preamble); recorded in the W&B config so a run's provenance
# says so explicitly rather than by omission.
TORCH_COMPILE = False
PRECISIONS = ("auto", "bf16", "fp16", "fp32")

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
    # Explicit in every configs/*.yaml (tests/test_train_precision.py enforces it) rather than
    # relying on whichever entity the local W&B login defaults to; None = W&B's default entity.
    wandb_entity: str | None = None


@dataclass
class EvalSection:
    eval_every: int = 500
    # Fixed seeded subset sizes for the periodic eval hook (spec §11: "E1[500], E2[300]" for the
    # main run; "keep it cheap and configurable ... smoke uses small subsets"). The official dev
    # set is always evaluated whole (150 sentences), never subset, so it has no size field here.
    e1_n: int = 500
    e2_n: int = 300
    e3_n: int = 300
    n_samples_table: int = 20


@dataclass
class TrainConfig:
    name: str
    group: str = "main"  # pilot | ablation | main | smoke (spec §11 W&B run groups)
    seed: int = 1234
    label_smoothing: float = 0.1
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"; see resolve_precision
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
    cfg = TrainConfig(**kwargs)
    if cfg.precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {cfg.precision!r}")
    return cfg


# ---------------------------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA). `warn_only=True` because a couple of ops used
    here (e.g. embedding backward with a padding_idx) don't have a deterministic CUDA kernel;
    warn rather than hard-fail so the same code path runs on both CPU (fully deterministic) and
    a future GPU run. On CUDA, also pins CUBLAS_WORKSPACE_CONFIG (cuBLAS reads it when its handle
    is first created, so this must run before any CUDA matmul) and collapses torch's per-call
    nondeterminism warning to one line per process.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # "once" (not "default"): "default" dedupes per call site, so the same op reached from many
    # modules/steps would still print repeatedly; "once" prints the first occurrence only.
    # Two real texts seen: generic ops ("... does not have a deterministic implementation") and
    # the memory-efficient SDPA backward ("... defaults to a non-deterministic algorithm").
    warnings.filterwarnings(
        "once",
        message=r".*(does not have a deterministic implementation|non-deterministic algorithm).*",
    )


# ---------------------------------------------------------------------------------------------
# Precision, TF32 and SDPA kernel report
# ---------------------------------------------------------------------------------------------


def resolve_precision(requested: str, device: torch.device) -> str:
    """Resolve the config's `precision` to one of "bf16" | "fp16" | "fp32".

    `auto`: bf16 on CUDA when `torch.cuda.is_bf16_supported()`, else fp16 on CUDA, fp32 on CPU.
    An explicit bf16/fp16 on a non-CUDA device is an error (CPU autocast would silently change
    the numerics the CPU tests pin down); explicit fp32 is always allowed.
    """
    if requested not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {requested!r}")
    if requested == "auto":
        if device.type != "cuda":
            return "fp32"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested in ("bf16", "fp16") and device.type != "cuda":
        raise ValueError(f"precision={requested!r} requires a CUDA device (got {device.type!r})")
    return requested


_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def autocast_context(device: torch.device, precision: str) -> contextlib.AbstractContextManager:
    """Autocast context for a resolved precision ("fp32" -> disabled)."""
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPES[precision])


CUDA_MEMORY_FRACTION = 0.95  # see the set_per_process_memory_fraction call in train()


def enable_tf32(device: torch.device) -> bool:
    """Enable TF32 for fp32 matmuls/convs on CUDA (Ampere+; harmless on older GPUs). Returns
    whether it was enabled. Under bf16/fp16 autocast only the remaining fp32 ops benefit.
    """
    if device.type != "cuda":
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return True


_SDPA_HEADER = re.compile(r"^(.*?) kernel not used because:", re.IGNORECASE)
_SDPA_TRACE = re.compile(r"\s*\(Triggered internally at [^)]*\)\.?")


def _flash_reason(messages: list[str]) -> str:
    """Pull the flash-attention part out of the warnings PyTorch emits when a single-backend
    `sdpa_kernel` context cannot run. The dispatcher's debug output lists every backend, each as
    a "<name> kernel not used because:" header followed by its reasons; only the reasons under
    the flash header describe flash (the others merely say "runtime disabled" by our restriction).
    """
    section: str | None = None
    reasons: list[str] = []
    for message in messages:
        text = _SDPA_TRACE.sub("", message.split("\n")[0]).strip()
        header = _SDPA_HEADER.match(text)
        if header:
            section = header.group(1).lower()
        elif section is not None and "flash" in section and text:
            reasons.append(text)
    return "; ".join(dict.fromkeys(reasons)) or "unsupported"


def probe_sdpa_kernel(
    model_cfg: ModelConfig, device: torch.device, precision: str, seq_len: int = 61
) -> dict[str, Any]:
    """Which SDPA backend do the model's real attention calls use on this GPU/dtype/mask?

    Replays the three attention call shapes of nmt/model/transformer.py with the dtype autocast
    produces, requires_grad inputs, training-time dropout and the exact boolean masks used there:
    encoder self-attn (B,1,1,Ts padding), decoder self-attn (B,1,T,T causal & padding) and
    cross-attn (B,1,1,Ts padding). `seq_len` is deliberately not a multiple of 8 because
    alignment is a common reason fused kernels decline. Each backend is tried alone, in the
    dispatcher's priority order (flash, efficient, cudnn, math), via `sdpa_kernel`; the first that
    runs a forward+backward is the one the dispatcher would pick. Returns {"kernel",
    "flash_unavailable_reason", "per_site", "summary"}; `summary` is the printed line.
    """
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if device.type != "cuda":
        return {
            "kernel": "n/a",
            "flash_unavailable_reason": None,
            "per_site": {},
            "summary": "SDPA kernel: n/a (not CUDA)",
        }
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    b, h = 2, model_cfg.n_heads
    dh = model_cfg.d_model // model_cfg.n_heads
    pad = torch.ones(b, seq_len, dtype=torch.bool, device=device)
    pad[:, -3:] = False  # a realistic padded tail
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    masks = {
        "encoder_self": pad[:, None, None, :],
        "decoder_self": causal[None, None] & pad[:, None, None, :],
        "cross": pad[:, None, None, :],
    }
    order = [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
    per_site: dict[str, str] = {}
    flash_reasons: dict[str, str] = {}
    for site, mask in masks.items():
        qkv = [
            torch.randn(b, h, seq_len, dh, device=device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]
        chosen = "none"
        for name, backend in order:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    with sdpa_kernel([backend]):
                        out = F.scaled_dot_product_attention(
                            *qkv, attn_mask=mask, dropout_p=model_cfg.dropout
                        )
                        out.sum().backward()
                except RuntimeError:
                    if name == "flash":
                        flash_reasons[site] = _flash_reason([str(w.message) for w in caught])
                    continue
            chosen = name
            break
        per_site[site] = chosen
    kernels = set(per_site.values())
    kernel = kernels.pop() if len(kernels) == 1 else "mixed"
    reason = "; ".join(dict.fromkeys(flash_reasons.values())) or None
    summary = f"SDPA kernel: {kernel}"
    if kernel == "mixed":
        summary += " (" + ", ".join(f"{k}={v}" for k, v in per_site.items()) + ")"
    if reason:
        summary += f" (flash unavailable: {reason})"
    return {
        "kernel": kernel,
        "flash_unavailable_reason": reason,
        "per_site": per_site,
        "summary": summary,
    }


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


def _git_dirty() -> bool | None:
    """True if tracked files differ from HEAD (untracked files are ignored: runs/, wandb/ and
    data artifacts are always untracked and would make every run "dirty"); None if unknown."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _nvidia_driver_version() -> str | None:
    """Driver version via nvidia-smi (pynvml is not a dependency); None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0].strip() or None
    except Exception:
        return None


def collect_run_provenance(
    cfg: TrainConfig,
    device: torch.device,
    precision: str,
    tf32: bool,
    sdpa: dict[str, Any],
    model: Transformer,
    shard_dir: Path,
) -> dict[str, Any]:
    """Provenance recorded in the W&B run config and `run_info.json` (spec §11): code state, the
    numeric setup that produced the run, hardware/software versions and the hashes of the data
    manifests. The platform string is `platform.platform()` (OS + version, no hostname).
    """
    on_cuda = device.type == "cuda"
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "precision": precision,
        "tf32": tf32,
        "torch_compile": TORCH_COMPILE,
        "sdpa_kernel": sdpa["kernel"],
        "sdpa_kernel_summary": sdpa["summary"],
        "sdpa_kernel_per_site": sdpa["per_site"],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if on_cuda else None,
        "gpu_total_memory_mb": (
            torch.cuda.get_device_properties(device).total_memory / (1024**2) if on_cuda else None
        ),
        "driver_version": _nvidia_driver_version() if on_cuda else None,
        "platform": platform.platform(),
        "param_count": model.param_count(),
        "data_manifest_sha256": _file_sha256(REPO_ROOT / "data" / "data_manifest.json"),
        "shard_manifest_sha256": _file_sha256(Path(shard_dir) / "manifest.json"),
    }


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


def _feed_hash(h: Any, obj: Any) -> None:
    """Feed `obj` (nested tuples/lists/dicts of tensors, arrays and scalars) into hash `h` in a
    canonical, type-tagged byte form, so equal states hash equal and a one-bit change does not."""
    if isinstance(obj, torch.Tensor):
        h.update(b"T" + str(obj.dtype).encode() + str(tuple(obj.shape)).encode())
        h.update(obj.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(b"A" + str(obj.dtype).encode() + str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        h.update(b"D%d" % len(obj))
        for key in sorted(obj, key=str):
            _feed_hash(h, key)
            _feed_hash(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L%d" % len(obj))
        for item in obj:
            _feed_hash(h, item)
    else:
        h.update(b"S" + repr(obj).encode())


def rng_fingerprint(rng: dict[str, Any], sampler_state: dict[str, Any]) -> str:
    """12-hex sha256 prefix over every saved RNG state (python, numpy, torch CPU, torch CUDA) and
    the sampler state. Printed at every save and again after a restore so a resume can be checked
    by eye (or by a notebook) for bit-exact state recovery."""
    h = hashlib.sha256()
    _feed_hash(h, rng)
    _feed_hash(h, sampler_state)
    return h.hexdigest()[:12]


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sampler: BucketedSampler,
    decay_start_step: int | None,
    grad_skip_count: int,
    precision: str | None = None,
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
        "precision": precision,  # resolved precision; resuming under a different one fails loudly
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
    # Windows tools (and `wandb login`) write `_netrc`, not `.netrc`; checking only `.netrc`
    # silently downgraded every local `--wandb online` run to offline.
    if os.environ.get("NETRC"):
        candidates = [Path(os.environ["NETRC"])]
    else:
        candidates = [Path.home() / ".netrc", Path.home() / "_netrc"]
    for netrc_path in candidates:
        if netrc_path.is_file():
            try:
                if "api.wandb.ai" in netrc_path.read_text(encoding="utf-8"):
                    return True
            except OSError:
                continue
    return False


def _resolve_wandb_mode(requested: str) -> str:
    if requested == "online" and not _wandb_authenticated():
        print(
            "wandb: no API key found (env WANDB_API_KEY / ~/.netrc); falling back to offline mode."
        )
        return "offline"
    return requested


def init_wandb(
    cfg: TrainConfig,
    mode: str,
    seed: int,
    model: Transformer,
    device: torch.device,
    provenance: dict[str, Any] | None = None,
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
            entity=cfg.logging.wandb_entity,
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
                # Last, so the resolved values (e.g. precision "bf16", not the config's "auto")
                # win over the same-named raw config keys.
                **(provenance or {}),
            },
        )
    except Exception as exc:  # noqa: BLE001 - W&B failures must never abort training
        print(f"wandb: init failed ({exc!r}); continuing without W&B logging.")
        return None


# ---------------------------------------------------------------------------------------------
# Eval hook (P4: nmt.evaluate.build_train_eval_fn builds the real one)
# ---------------------------------------------------------------------------------------------

# The third parameter is the active W&B run (or None) -- a hook that logs a W&B Table (e.g. the
# spec §11 sample-translations table) needs the run object directly, since a Table isn't
# JSON-serializable and so can't travel through the returned dict (which IS written verbatim to
# metrics.jsonl via json.dumps below).
EvalFn = Callable[[nn.Module, int, Any], dict[str, float]]


def default_eval_fn(model: nn.Module, step: int, wandb_run: Any = None) -> dict[str, float]:
    """No-op eval hook placeholder (used when the caller doesn't pass one, e.g. most tests).
    `nmt.evaluate.build_train_eval_fn` builds the real spec §8/§11 greedy BLEU/chrF hook.
    Validation loss on E1 is computed directly in the train loop below (`_eval_val_loss`), not
    through this hook, since it's needed for checkpoint-selection sanity independent of it.
    """
    del model, step, wandb_run
    return {}


@torch.no_grad()
def _eval_val_loss(
    model: Transformer,
    eval_dataset: ShardDataset,
    cfg: TrainConfig,
    device: torch.device,
    max_batches: int = 20,
    precision: str = "fp32",
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
        # Same autocast as the train step, so val loss is comparable with the train loss.
        with autocast_context(device, precision):
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


def _last_logged_losses(metrics_path: Path, upto_step: int, n: int = 3) -> list[float]:
    """Last `n` per-step losses logged at or before `upto_step` (a step can appear twice if an
    earlier run died after logging it but before checkpointing; the last occurrence wins)."""
    by_step: dict[int, float] = {}
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            if "step" in row and "loss" in row and "eval" not in row and row["step"] <= upto_step:
                by_step[row["step"]] = round(row["loss"], 6)
    return [by_step[k] for k in sorted(by_step)][-n:]


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
    stop_after_first_ckpt: bool = False,
) -> None:
    """Run (or resume) training for `cfg`. Every call builds a brand-new model/optimizer/sampler
    from scratch and, if `resume=True`, restores them from the latest checkpoint in
    `cfg.ckpt.dir` — this mirrors a real process restart exactly (no in-memory state survives
    between calls), which is what makes the resume-determinism test meaningful.

    `stop_after_first_ckpt` (forced-resume testing): a FRESH invocation returns right after its
    first checkpoint save; a resumed invocation ignores the flag. The CKPT_SAVED / RESUMED /
    RESUME_CONTEXT / POST_RESUME / STOPPED_AFTER_FIRST_CKPT lines printed below are a fixed
    contract that the Colab notebook greps; do not reword them.
    """
    seed = cfg.seed if seed is None else seed
    device = _resolve_device(cfg)
    seed_everything(seed)
    precision = resolve_precision(cfg.precision, device)
    tf32 = enable_tf32(device)
    if device.type == "cuda":
        # Windows/WDDM spills over-budget allocations into shared system RAM instead of raising
        # OOM, which silently turns a too-large micro-batch into a many-times-slower run. Capping
        # the caching allocator makes that case fail loudly instead (same cap as the probe).
        torch.cuda.set_per_process_memory_fraction(
            CUDA_MEMORY_FRACTION, device.index if device.index is not None else 0
        )
    print(f"precision: {precision} (config: {cfg.precision}), tf32: {tf32}, device: {device}")

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
    # Loss scaling is only needed for fp16's narrow exponent range; bf16 shares fp32's range.
    scaler = torch.amp.GradScaler(device="cuda", enabled=(precision == "fp16"))

    step = 0
    decay_start_step: int | None = None
    grad_skip_count = 0
    resumed_ckpt: dict[str, Any] | None = None
    if resume:
        resumed_ckpt = load_latest_checkpoint(ckpt_dir)
        if resumed_ckpt is not None:
            saved_precision = resumed_ckpt.get("precision")
            if saved_precision is not None and saved_precision != precision:
                raise RuntimeError(
                    f"cannot resume: checkpoint was trained with precision={saved_precision!r} "
                    f"but this run resolved precision={precision!r} (config {cfg.precision!r}). "
                    "Mixing precisions mid-run breaks the loss-scale/optimizer-state contract; "
                    "pass the original --precision or start a fresh run in a new ckpt dir."
                )
            model.load_state_dict(resumed_ckpt["model"])
            optimizer.load_state_dict(resumed_ckpt["optimizer"])
            scaler.load_state_dict(resumed_ckpt["scaler"])
            sampler.load_state_dict(resumed_ckpt["sampler"])
            step = resumed_ckpt["step"]
            decay_start_step = resumed_ckpt.get("decay_start_step")
            grad_skip_count = resumed_ckpt.get("grad_skip_count", 0)
    if cooldown_now and decay_start_step is None:
        decay_start_step = step

    # fork_rng: the probe's random inputs and dropout would otherwise advance the CUDA RNG that
    # training (and the RNG fingerprint) depends on.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        sdpa = probe_sdpa_kernel(_build_model_config(cfg), device, precision)
    print(sdpa["summary"])
    provenance = collect_run_provenance(cfg, device, precision, tf32, sdpa, model, shard_dir)
    provenance["precision_requested"] = cfg.precision
    (run_dir / "run_info.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    wandb_run = init_wandb(cfg, wandb_mode, seed, model, device, provenance)

    target_steps = max_steps if max_steps is not None else cfg.optim.planned_steps
    eval_dataset: ShardDataset | None = None
    eval_disabled_reason: str | None = None
    try:
        eval_dataset = ShardDataset(shard_dir, cfg.data.eval_split)
    except FileNotFoundError as exc:
        # Non-fatal (a missing eval split must not abort training), but silent-suppression here
        # previously hid validation entirely with no trace anywhere it ran — loud print plus a
        # durable record in both metrics.jsonl and the W&B run config, so absence is visible.
        eval_disabled_reason = (
            f"eval split {cfg.data.eval_split!r} not found under {shard_dir}: {exc}"
        )
        print(f"WARNING: {eval_disabled_reason}; validation loss/eval_fn disabled this run.")
    if eval_disabled_reason is not None and wandb_run is not None:
        with contextlib.suppress(Exception):  # W&B failures must never abort training
            wandb_run.config.update(
                {"eval_disabled_reason": eval_disabled_reason}, allow_val_change=True
            )

    resumed = resumed_ckpt is not None
    if resumed_ckpt is not None:
        # Restore RNG as late as possible (after W&B init, the SDPA probe, dataset construction),
        # so nothing between restore and the first training step can consume it, then fingerprint
        # the LIVE state against what the checkpoint holds.
        _restore_rng_state(resumed_ckpt["rng"])
        saved_fp = rng_fingerprint(resumed_ckpt["rng"], resumed_ckpt["sampler"])
        live_fp = rng_fingerprint(_rng_state(), sampler.state_dict())
        print(f"RESUMED step={step} rng_fingerprint={live_fp} matches_saved={live_fp == saved_fp}")
        print(f"RESUME_CONTEXT last_losses={_last_logged_losses(metrics_path, step)}")
    if stop_after_first_ckpt and resumed:
        print("stop-after-first-ckpt ignored: resumed run")
    post_resume_losses: list[float] = []
    resume_step = step

    micro_iter = _iter_micro_batches(sampler)
    last_ckpt_time = time.monotonic()
    start_time = time.monotonic()
    model.train()

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        if eval_disabled_reason is not None:
            metrics_file.write(json.dumps({"eval_disabled_reason": eval_disabled_reason}) + "\n")
            metrics_file.flush()
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
                with autocast_context(device, precision):
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
            elif step % cfg.logging.log_every == 0:
                print(
                    f"step {step} loss={step_loss:.4f} lr={lr:.2e} tok/s={tok_per_sec:.0f} "
                    f"step_time={time.monotonic() - step_start:.2f}s peak_mem={gpu_mem_mb:.0f}MB",
                    flush=True,
                )

            if resumed and len(post_resume_losses) < 3 and step > resume_step:
                post_resume_losses.append(round(step_loss, 6))
                if len(post_resume_losses) == 3:
                    print(f"POST_RESUME losses={post_resume_losses}")

            if step % cfg.logging.log_every == 0 and wandb_run is not None:
                wandb_run.log(row, step=step)

            eval_due = eval_dataset is not None and cfg.eval.eval_every > 0
            if eval_due and step % cfg.eval.eval_every == 0:
                val_loss = _eval_val_loss(
                    model,
                    eval_dataset,  # type: ignore[arg-type]
                    cfg,
                    device,
                    precision=precision,
                )
                # Autocast around the whole hook so its greedy decoding runs in the training
                # precision too (the hook lives in nmt.evaluate and has no precision knob).
                with autocast_context(device, precision):
                    extra = eval_fn(model, step, wandb_run)
                eval_row = {"step": step, "val_loss": val_loss, **extra}
                metrics_file.write(json.dumps({"eval": eval_row}) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    log_row = {f"eval/{k}": v for k, v in eval_row.items() if k != "step"}
                    wandb_run.log(log_row, step=step)

            due_by_time = (time.monotonic() - last_ckpt_time) / 60.0 >= cfg.ckpt.ckpt_minutes
            due_by_steps = cfg.ckpt.ckpt_steps is not None and step % cfg.ckpt.ckpt_steps == 0
            if due_by_time or due_by_steps or step >= target_steps:
                # Fingerprint the state save_checkpoint is about to capture (nothing between here
                # and its `_rng_state()` call consumes RNG).
                fp = rng_fingerprint(_rng_state(), sampler.state_dict())
                ckpt_path = save_checkpoint(
                    ckpt_dir,
                    step,
                    model,
                    optimizer,
                    scaler,
                    sampler,
                    decay_start_step,
                    grad_skip_count,
                    precision=precision,
                )
                print(f"CKPT_SAVED step={step} rng_fingerprint={fp} path={ckpt_path}", flush=True)
                prune_checkpoints(
                    ckpt_dir, cfg.ckpt.keep_last, decay_start_step, cfg.ckpt.keep_decay_phase
                )
                last_ckpt_time = time.monotonic()
                if stop_after_first_ckpt and not resumed:
                    print(f"STOPPED_AFTER_FIRST_CKPT step={step} path={ckpt_path}", flush=True)
                    break

    if resumed and 0 < len(post_resume_losses) < 3:
        print(f"POST_RESUME losses={post_resume_losses}")  # run ended before 3 post-resume steps

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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap this invocation's step count below optim.planned_steps (the WSD schedule "
        "itself still targets planned_steps; use --planned-steps to change that).",
    )
    parser.add_argument("--wandb", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument(
        "--cooldown-now", action="store_true", help="Start WSD decay at the current step."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Train on a generated copy-task instead of real shards.",
    )
    parser.add_argument(
        "--run-dir", default=None, help="Override logging.run_dir (metrics.jsonl location)."
    )
    parser.add_argument("--ckpt-dir", default=None, help="Override ckpt.dir (checkpoint location).")
    parser.add_argument(
        "--data-dir", default=None, help="Override data.shard_dir (where train/eval shards live)."
    )
    parser.add_argument(
        "--planned-steps",
        type=int,
        default=None,
        help="Override optim.planned_steps (the WSD schedule's total step budget), e.g. from "
        "the pilot's scripts/plan_steps.py recommendation.",
    )
    parser.add_argument(
        "--precision",
        choices=list(PRECISIONS),
        default=None,
        help="Override the config's top-level `precision` (auto|bf16|fp16|fp32).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=None, help="Override config.device."
    )
    parser.add_argument(
        "--ckpt-steps",
        type=int,
        default=None,
        help="Override ckpt.ckpt_steps (save every N steps).",
    )
    parser.add_argument(
        "--stop-after-first-ckpt",
        action="store_true",
        help="Forced-resume test: a FRESH run exits 0 right after its first checkpoint save "
        "(prints STOPPED_AFTER_FIRST_CKPT); a resumed run ignores this flag.",
    )
    return parser.parse_args(argv)


def build_eval_fn(cfg: TrainConfig, seed: int) -> EvalFn:
    """The real spec §11 periodic eval hook (greedy BLEU/chrF + W&B sample table), or the no-op
    placeholder when the committed tokenizer is absent. Lives here, not only in nmt.pipeline,
    so `python -m nmt.train` (what the Colab notebook runs) logs the BLEU/chrF curves too.

    The hook decodes on the TRAINING device: it builds its input tensors on `device` and uses
    the model in place, so a CPU device with a CUDA model would crash, and CPU greedy decoding
    of ~1.2k sentences with the 50M model on Colab's 2 vCPUs would eat the GPU budget.
    """
    tokenizer_path = Path(__file__).resolve().parents[1] / "tokenizer" / "spm.model"
    if not tokenizer_path.is_file():
        print(f"WARNING: {tokenizer_path} missing; BLEU/chrF eval hook disabled this run.")
        return default_eval_fn
    from nmt.evaluate import TrainEvalConfig, build_train_eval_fn  # lazy: heavy imports

    return build_train_eval_fn(
        TrainEvalConfig(
            tokenizer_path=tokenizer_path,
            e1_n=cfg.eval.e1_n,
            e2_n=cfg.eval.e2_n,
            e3_n=cfg.eval.e3_n,
            n_samples_table=cfg.eval.n_samples_table,
            seed=seed if seed is not None else cfg.seed,
            device=_resolve_device(cfg).type,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    if args.run_dir is not None:
        cfg.logging.run_dir = args.run_dir
    if args.ckpt_dir is not None:
        cfg.ckpt.dir = args.ckpt_dir
    if args.data_dir is not None:
        cfg.data.shard_dir = args.data_dir
    if args.planned_steps is not None:
        cfg.optim.planned_steps = args.planned_steps
    if args.precision is not None:
        cfg.precision = args.precision
    if args.device is not None:
        cfg.device = args.device
    if args.ckpt_steps is not None:
        cfg.ckpt.ckpt_steps = args.ckpt_steps
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
        stop_after_first_ckpt=args.stop_after_first_ckpt,
        eval_fn=default_eval_fn if args.synthetic else build_eval_fn(cfg, args.seed),
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
