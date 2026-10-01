from __future__ import annotations

# GPU micro-batch probe (spec §14 P5): on CUDA, builds the main architecture (spec §5 / configs/
# main.yaml's model section) and tries increasing micro-batch token counts with fp16 autocast
# forward+backward until CUDA OOM or a hard cap, then reports the largest that fits with a memory
# headroom margin. Run first by the pilot notebook cell (CONFIG == "pilot") so later configs'
# `batch.max_tokens` can be set from a measured value instead of a guess.
#
# CPU has no GPU memory budget to probe: `main()` is a deliberate, non-raising no-op off CUDA, so
# it is safe to import/call from the CPU smoke notebook run and from `uv run pytest` without ever
# touching a GPU.
#
# CLI: `python scripts/probe_microbatch.py [--out runs/pilot/probe.json] [--headroom 0.15]
#   [--seq-len 64]`
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nmt.model.transformer import ModelConfig, Transformer

DEFAULT_HEADROOM = 0.15  # spec §14 P5: "reports the max that fits with 15% headroom" -- real
# training also carries dataloader/CUDA-context/fragmentation overhead beyond a single probed
# forward+backward, so the reported safe max deliberately undershoots the raw OOM boundary.
DEFAULT_CANDIDATE_TOKEN_COUNTS = (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
DEFAULT_SEQ_LEN = 64  # synthetic sequence length per example; token_count // seq_len -> batch size


@dataclass
class ProbeResult:
    device_name: str
    total_memory_mb: float
    headroom: float
    candidates_tried: list[int]
    max_fitting_tokens: int | None
    max_fitting_with_headroom_tokens: int | None
    peak_memory_at_max_mb: float | None


def build_main_model(vocab_size: int = 16000) -> Transformer:
    """Main architecture (spec §5 / configs/main.yaml): 8 enc / 4 dec, d=512, 8 heads, FFN 2048."""
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
    model: Transformer, token_count: int, seq_len: int, device: torch.device
) -> float:
    """One fp16 forward+backward on a synthetic (batch, seq_len) batch sized so that
    `batch_size * seq_len` is approximately `token_count` (batch_size at least 1). Returns peak
    CUDA memory in MB for this step. Propagates `torch.cuda.OutOfMemoryError` if it doesn't fit.
    """
    batch_size = max(1, token_count // seq_len)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    src = torch.randint(4, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    tgt_in = torch.randint(4, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    tgt_out = torch.randint(4, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(src, tgt_in)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
        )
    loss.backward()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def probe(
    candidates: tuple[int, ...] = DEFAULT_CANDIDATE_TOKEN_COUNTS,
    seq_len: int = DEFAULT_SEQ_LEN,
    headroom: float = DEFAULT_HEADROOM,
    device: torch.device | None = None,
) -> ProbeResult:
    """Try `candidates` (ascending) until CUDA OOM or the cap; report the max that fit at all and
    the max that fit within `headroom` of total device memory. Requires CUDA (raises otherwise --
    callers on CPU should check `torch.cuda.is_available()` first, as `main()` below does).
    """
    device = torch.device("cuda") if device is None else device
    if device.type != "cuda":
        raise RuntimeError("probe_microbatch.probe requires a CUDA device")

    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    model = _build_or_reuse_model(device)

    max_fitting: int | None = None
    peak_at_max: float | None = None
    for tokens in sorted(candidates):
        try:
            peak_mb = _try_one_step(model, tokens, seq_len, device)
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
        total_memory_mb=total_mb,
        headroom=headroom,
        candidates_tried=sorted(candidates),
        max_fitting_tokens=max_fitting,
        max_fitting_with_headroom_tokens=max_with_headroom,
        peak_memory_at_max_mb=peak_at_max,
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
        description="Probe the largest fp16 micro-batch (in tokens) that fits on this GPU for the "
        "main architecture, with a memory headroom margin (spec §14 P5)."
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write probe.json.")
    parser.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "probe_microbatch: no CUDA device available; skipping "
            "(CPU has no GPU memory budget to probe)."
        )
        return 0
    result = probe(headroom=args.headroom, seq_len=args.seq_len)
    print(f"device: {result.device_name} ({result.total_memory_mb:.0f} MB total)")
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
