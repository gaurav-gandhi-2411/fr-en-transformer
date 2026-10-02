from __future__ import annotations

# Reproduces the per-step token figures recorded in PLAN.md ("Batch-size definition"): padded and
# real (non-pad) source + target-input tokens per optimizer step from reports/epoch_accounting.json,
# and the real TARGET-side tokens per step, which that JSON does not store. The target share comes
# from the train shards: per example the source side has len + 1 real tokens (EOS) and the
# target-input side has len + 1 (BOS), the same basis as `nmt.data.loader.batch_token_count`.
# For concat_prob 0.0 the target-side figure is exact; for concat_prob > 0 it assumes the same
# source/target split as the un-concatenated epoch (not measured separately).
#
# CLI: `python scripts/token_arithmetic.py [--shard-dir data/shards]
#   [--accounting reports/epoch_accounting.json]`
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def shard_real_tokens(shard_dir: Path) -> tuple[int, int, int]:
    """(source real incl. EOS, target-input real incl. BOS, examples) over every train shard."""
    src_real = tgt_real = n = 0
    for shard in sorted((shard_dir / "train").glob("shard_*.npz")):
        with np.load(shard) as z:
            src_off, tgt_off = z["src_off"], z["tgt_off"]
        k = len(src_off) - 1
        src_real += int(src_off[-1]) + k
        tgt_real += int(tgt_off[-1]) + k
        n += k
    return src_real, tgt_real, n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=REPO / "data" / "shards")
    parser.add_argument(
        "--accounting", type=Path, default=REPO / "reports" / "epoch_accounting.json"
    )
    args = parser.parse_args(argv)
    acc = json.loads(args.accounting.read_text(encoding="utf-8"))
    for r in acc["epoch0_replays"]:
        steps = r["optimizer_steps_per_epoch"]
        print(
            f"max_tokens={r['max_tokens']} concat={r['concat_prob']}: "
            f"padded/step={r['padded_tokens_src_plus_tgt']:,}/{steps:,}"
            f"={r['padded_tokens_src_plus_tgt'] / steps:,.1f} "
            f"real(src+tgt_in)/step={r['real_tokens_src_plus_tgt']:,}/{steps:,}"
            f"={r['real_tokens_src_plus_tgt'] / steps:,.1f} "
            f"micro/step={r['micro_batches'] / steps:.3f} padding_share={r['padding_share']:.4f}"
        )
    src_real, tgt_real, n = shard_real_tokens(args.shard_dir)
    total = src_real + tgt_real
    print(f"shards: pairs={n:,} src_real={src_real:,} tgt_in_real={tgt_real:,} sum={total:,}")
    share = tgt_real / (src_real + tgt_real)
    print(f"target share of real src+tgt_in tokens = {share:.6f}")
    for r in acc["epoch0_replays"]:
        steps = r["optimizer_steps_per_epoch"]
        if r["concat_prob"] == 0.0:
            print(
                f"EXACT target tokens/step (concat 0.0, max_tokens {r['max_tokens']}): "
                f"{tgt_real:,} / {steps:,} = {tgt_real / steps:,.1f}"
            )
        else:
            est = share * r["real_tokens_src_plus_tgt"] / steps
            print(
                f"ESTIMATE target tokens/step (concat {r['concat_prob']}, max_tokens "
                f"{r['max_tokens']}): {share:.6f} x {r['real_tokens_src_plus_tgt']:,} / {steps:,} "
                f"= {est:,.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
