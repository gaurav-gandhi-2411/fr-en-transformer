from __future__ import annotations

# Epoch accounting on the trainer's own token basis. `nmt.data.loader.batch_token_count` counts
# src.numel() + tgt_in.numel(): BOTH sides, padded (batch_size x padded length, incl. EOS on src
# and BOS on tgt_in), and `nmt.train` accumulates micro-batches until that sum >= tokens_per_step.
# So epochs depend on padding, i.e. on the micro-batch budget and on concat augmentation (which
# runs AFTER bucketing and so can inflate a batch's padded length). This replays the real
# BucketedSampler for one full epoch per (max_tokens, concat_prob) setting and records padded and
# real (non-pad) tokens, examples, micro-batches, and optimizer steps per epoch, plus the
# distribution of padded micro-batch sizes (the memory-relevant quantity).
#
# CLI: `python -m scripts.epoch_accounting --out reports/epoch_accounting.json
#   [--setting 4096:0.0 --setting 4096:0.15 ...] [--tokens-per-step 25000] [--seed 1234]`
import argparse
import json
import time
from pathlib import Path

import numpy as np

from nmt.data.loader import BucketedSampler, ShardDataset, batch_token_count

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ("4096:0.0", "4096:0.15", "8192:0.0", "8192:0.15")


def replay_epoch(
    dataset: ShardDataset, max_tokens: int, concat_prob: float, seed: int, tokens_per_step: int
) -> dict:
    sampler = BucketedSampler(
        dataset, max_tokens=max_tokens, seed=seed, concat_prob=concat_prob, concat_max_len=256
    )
    padded, real, examples, steps, acc = 0, 0, 0, 0, 0
    sizes: list[int] = []
    t0 = time.monotonic()
    for batch in sampler:
        n = batch_token_count(batch)
        sizes.append(n)
        padded += n
        real += int((batch.src != 0).sum()) + int((batch.tgt_in != 0).sum())
        examples += batch.src.shape[0]
        acc += n
        if acc >= tokens_per_step:  # same rule as nmt.train's accumulation loop
            steps += 1
            acc = 0
    arr = np.array(sizes)
    return {
        "max_tokens": max_tokens,
        "concat_prob": concat_prob,
        "micro_batches": len(sizes),
        "examples": examples,
        "padded_tokens_src_plus_tgt": padded,
        "real_tokens_src_plus_tgt": real,
        "padding_share": 1 - real / padded,
        "optimizer_steps_per_epoch": steps,
        "partial_step_tokens_left": acc,
        "padded_microbatch_tokens": {
            "median": float(np.median(arr)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "n_over_budget": int((arr > max_tokens * 2).sum()),
            "n_over_4x_budget": int((arr > max_tokens * 4).sum()),
        },
        "replay_seconds": round(time.monotonic() - t0, 1),
    }


# (label, max_tokens, concat_prob, optimizer steps, arithmetic for the step count)
MAIN_STEPS = 24645
ABLATION_3070_STEPS = 2889
L4_FALLBACK_STEPS = int(40 * 60 * 0.85 / 0.4966)  # 40 min budget x 0.85 utilisation / s per step
EPOCH_PLANS = (
    ("main_l4", 4096, 0.15, MAIN_STEPS, "main: 24,645 steps (L4, micro-batch 4096)"),
    ("3070_s1_s2", 8192, 0.0, ABLATION_3070_STEPS, "3070 ablation S1/S2: 2,889 steps"),
    ("3070_s3", 8192, 0.15, ABLATION_3070_STEPS, "3070 ablation S3: 2,889 steps"),
    ("l4_fallback_s1_s2", 4096, 0.0, L4_FALLBACK_STEPS, "L4 fallback S1/S2: floor(2040/0.4966)"),
    ("l4_fallback_s3", 4096, 0.15, L4_FALLBACK_STEPS, "L4 fallback S3: floor(2040/0.4966)"),
)


def epochs_block(replays: list[dict]) -> dict:
    """Epochs = planned optimizer steps / optimizer_steps_per_epoch, per planned run, looked up
    from the replay of the matching (max_tokens, concat_prob) setting. Missing setting -> skipped.
    """
    by_setting = {(r["max_tokens"], r["concat_prob"]): r for r in replays}
    out: dict[str, dict] = {}
    for label, mt, cp, steps, note in EPOCH_PLANS:
        r = by_setting.get((mt, cp))
        if r is None:
            continue
        spe = r["optimizer_steps_per_epoch"]
        out[label] = {
            "note": note,
            "max_tokens": mt,
            "concat_prob": cp,
            "planned_steps": steps,
            "optimizer_steps_per_epoch": spe,
            "epochs": steps / spe,
            "arithmetic": f"{steps} / {spe} = {steps / spe:.3f}",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--setting", action="append", default=None)
    parser.add_argument("--tokens-per-step", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shard-dir", type=Path, default=REPO / "data" / "shards")
    args = parser.parse_args(argv)
    dataset = ShardDataset(args.shard_dir, "train")
    results = []
    for setting in args.setting or DEFAULT_SETTINGS:
        mt, cp = setting.split(":")
        r = replay_epoch(dataset, int(mt), float(cp), args.seed, args.tokens_per_step)
        print(json.dumps(r), flush=True)
        results.append(r)
    out = {
        "basis": "nmt.data.loader.batch_token_count = src.numel() + tgt_in.numel() (padded, both "
        "sides); nmt.train accumulates micro-batches until >= tokens_per_step",
        "tokens_per_step": args.tokens_per_step,
        "seed": args.seed,
        "n_train_pairs": len(dataset),
        "epoch0_replays": results,
        "epochs": epochs_block(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
