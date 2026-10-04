from __future__ import annotations

# GPU micro-batch probe: on CUDA, builds the main architecture (configs/
# main.yaml's model section) and tries increasing micro-batch token counts with autocast at the
# training precision (default `auto` = bf16 on the 3070, fp16 on a T4; same resolution and TF32 /
# deterministic-cuBLAS setup as nmt.train, so the measured memory matches a real run)
# forward+backward until CUDA OOM or a hard cap, then reports the largest that fits with a memory
# headroom margin. Run first by the pilot notebook cell (CONFIG == "pilot") so later configs'
# `batch.max_tokens` can be set from a measured value instead of a guess.
#
# CPU has no GPU memory budget to probe: `main()` is a deliberate, non-raising no-op off CUDA, so
# it is safe to import/call from the CPU smoke notebook run and from `uv run pytest` without ever
# touching a GPU.
#
# CLI (from the repo root; `-m` puts the repo root on sys.path so this also works from
# envs/cuda, where the repo is not installed as a package):
#   `uv run --project envs/cuda python -m scripts.probe_microbatch --out runs/pilot_3070/probe.json
#   [--precision auto|bf16|fp16|fp32] [--headroom 0.15] [--seq-len 64]`
# (`python scripts/probe_microbatch.py ...` also works: the repo root is added to sys.path below.)
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # direct-script invocation puts scripts/, not the root, first
    sys.path.insert(0, str(_REPO_ROOT))

from nmt.model.transformer import ModelConfig, Transformer  # noqa: E402 - needs the path above
from nmt.train import (  # noqa: E402
    autocast_context,
    cap_cuda_allocator,
    enable_tf32,
    label_smoothed_nll_loss,
    resolve_precision,
    seed_everything,
)

DEFAULT_HEADROOM = 0.15  # "reports the max that fits with 15% headroom" -- real
# training also carries dataloader/CUDA-context/fragmentation overhead beyond a single probed
# forward+backward, so the reported safe max deliberately undershoots the raw OOM boundary.
DEFAULT_CANDIDATE_TOKEN_COUNTS = (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
DEFAULT_SEQ_LEN = 64  # synthetic sequence length per example; token_count // seq_len -> batch size
# The allocator is capped exactly as in training (nmt.train.cap_cuda_allocator). Uncapped on
# Windows/WDDM, the probe "fit" 16384 tokens at an 8493 MB peak on an 8192 MB card, because the
# driver spills to system RAM instead of raising OOM.


@dataclass
class ProbeResult:
    device_name: str
    precision: str
    tf32: bool
    total_memory_mb: float
    headroom: float
    candidates_tried: list[int]
    max_fitting_tokens: int | None
    max_fitting_with_headroom_tokens: int | None
    peak_memory_at_max_mb: float | None
    memory_fraction: float
    includes_optimizer_step: bool = True


def build_main_model(vocab_size: int = 16000) -> Transformer:
    """Main architecture (configs/main.yaml): 8 enc / 4 dec, d=512, 8 heads, FFN 2048."""
    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        enc_layers=8,
        dec_layers=4,
        d_ff=2048,
        dropout=0.1,
        pos="rope",
    )
    return Transformer(cfg)


def _try_one_step(
    model: Transformer,
    token_count: int,
    seq_len: int,
    device: torch.device,
    precision: str = "fp16",
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One forward+backward at `precision`'s autocast on a synthetic (batch, seq_len) batch sized
    so that `batch_size * seq_len` is approximately `token_count` (batch_size at least 1). Returns
    peak CUDA memory in MB for this step. Propagates `torch.cuda.OutOfMemoryError` if it doesn't
    fit.
    """
    batch_size = max(1, token_count // seq_len)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    src = torch.randint(4, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    tgt_in = torch.randint(4, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    tgt_out = torch.randint(4, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    model.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        logits = model(src, tgt_in)
        # The real training loss, not plain cross_entropy: label smoothing keeps extra fp32
        # (tokens x vocab) intermediates alive through backward. Probing with cross_entropy
        # passed 12288 tokens, and the real pilot then OOMed on a 750 MiB allocation.
        loss = label_smoothed_nll_loss(logits, tgt_out, 0, 0.1)
    loss.backward()
    if optimizer is not None:
        # AdamW's two fp32 moment buffers (~400 MB for the 50M model) exist during real training;
        # a forward+backward-only probe would under-count peak memory by that much.
        optimizer.step()
    torch.cuda.synchronize(device)
    # Reserved, not allocated: reserved is what occupies VRAM and counts against the cap.
    return torch.cuda.max_memory_reserved(device) / (1024**2)


def probe(
    candidates: tuple[int, ...] = DEFAULT_CANDIDATE_TOKEN_COUNTS,
    seq_len: int = DEFAULT_SEQ_LEN,
    headroom: float = DEFAULT_HEADROOM,
    device: torch.device | None = None,
    precision: str = "auto",
) -> ProbeResult:
    """Try `candidates` (ascending) until CUDA OOM or the cap; report the max that fit at all and
    the max that fit within `headroom` of total device memory. Requires CUDA (raises otherwise --
    callers on CPU should check `torch.cuda.is_available()` first, as `main()` below does).
    """
    device = torch.device("cuda") if device is None else device
    if device.type != "cuda":
        raise RuntimeError("probe_microbatch.probe requires a CUDA device")

    resolved = resolve_precision(precision, device)
    tf32 = enable_tf32(device)
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    memory_fraction = cap_cuda_allocator(device)
    model = _build_or_reuse_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    max_fitting: int | None = None
    peak_at_max: float | None = None
    for tokens in sorted(candidates):
        try:
            peak_mb = _try_one_step(model, tokens, seq_len, device, resolved, optimizer)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        max_fitting = tokens
        peak_at_max = peak_mb
        if peak_mb >= total_mb * (1.0 - headroom):
            break  # already past the headroom margin; larger candidates can only be worse

    max_with_headroom: int | None = None
    if max_fitting is not None and peak_at_max is not None:
        if peak_at_max <= total_mb * (1.0 - headroom):
            max_with_headroom = max_fitting
        else:
            # The largest fitting candidate already eats into the headroom margin: step back to
            # the previous (smaller) candidate rather than report a figure with too little margin.
            tried_up_to_max = [c for c in sorted(candidates) if c <= max_fitting]
            max_with_headroom = tried_up_to_max[-2] if len(tried_up_to_max) >= 2 else None

    return ProbeResult(
        device_name=torch.cuda.get_device_name(device),
        precision=resolved,
        tf32=tf32,
        total_memory_mb=total_mb,
        headroom=headroom,
        candidates_tried=sorted(candidates),
        max_fitting_tokens=max_fitting,
        max_fitting_with_headroom_tokens=max_with_headroom,
        peak_memory_at_max_mb=peak_at_max,
        memory_fraction=memory_fraction,
    )


def _build_or_reuse_model(device: torch.device) -> Transformer:
    return build_main_model().to(device)


def write_result(result: ProbeResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the largest micro-batch (in tokens) that fits on this GPU for the "
        "main architecture at the training precision, with a memory headroom margin."
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write probe.json.")
    parser.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Autocast precision to probe (auto: bf16 where supported, else fp16).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "probe_microbatch: no CUDA device available; skipping "
            "(CPU has no GPU memory budget to probe)."
        )
        return 0
    seed_everything(1234)  # same deterministic-cuBLAS setup as training, so memory is comparable
    result = probe(headroom=args.headroom, seq_len=args.seq_len, precision=args.precision)
    print(f"device: {result.device_name} ({result.total_memory_mb:.0f} MB total)")
    print(f"precision: {result.precision}, tf32: {result.tf32}")
    print(
        f"max fitting (no margin): {result.max_fitting_tokens} tokens, "
        f"peak {result.peak_memory_at_max_mb:.1f} MB"
    )
    print(
        f"max fitting with {result.headroom:.0%} headroom: "
        f"{result.max_fitting_with_headroom_tokens} tokens"
    )
    if args.out is not None:
        write_result(result, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
