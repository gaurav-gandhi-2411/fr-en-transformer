from __future__ import annotations

# Pilot post-processing (spec §14 P5): reads a pilot run's metrics.jsonl, derives steady-state
# throughput (median tokens/s, seconds/optimizer-step over warmed-up steps) and peak GPU memory,
# then recommends `planned_steps` for each later run (S1/S2/S3: 40 min; main: ~4h; spec §1) with a
# safety margin reserved for periodic eval/checkpoint overhead -- which the pilot itself never
# pays, since configs/pilot.yaml sets `eval_every` high enough that it never fires. Writes
# `<metrics_dir>/plan.json` (or --out) and prints the exact PLANNED_STEPS to paste into the Colab
# notebook's parameters cell for each later run.
#
# CLI: `python scripts/plan_steps.py --metrics runs/pilot/metrics.jsonl
#   [--out runs/pilot/plan.json] [--tokens-per-step 25000] [--warmup-steps 20]
#   [--safety-margin 0.85] [--profile colab|3070]`
import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TOKENS_PER_STEP = 25000  # spec §6: ~25k target tokens per optimizer step, shared by every
# ablation/main config (configs/{s1_sin,s2_rope,s3_rope_concat,main}.yaml all set this exactly).
DEFAULT_WARMUP_STEPS = 20  # excluded from the steady-state throughput measurement: the first few
# steps pay for cuDNN/kernel autotune, cold caches and (on Colab) Drive-mount contention, and are
# not representative of the run's real sustained throughput.
DEFAULT_SAFETY_MARGIN = 0.85  # reserves 15% of the wall-clock budget for periodic eval/checkpoint
# I/O that the pilot itself does not pay for (see module docstring).

# Wall-clock budget (minutes) per later run, spec §1.
BUDGETS_MINUTES: dict[str, float] = {
    "s1_sin": 40.0,
    "s2_rope": 40.0,
    "s3_rope_concat": 40.0,
    "main": 240.0,
}

# Local RTX 3070 profile (`--profile 3070`): same 40-min ablation budget, planned from the
# pilot_3070 run's throughput, so s1/s2/s3_3070 all get one identical planned_steps and complete
# their WSD decay (their max_minutes is only a safety cap). main_ext_3070 is "~2x planned tokens"
# of main, i.e. 2x main's 240-minute budget. Do NOT run main_ext without GG approval.
BUDGETS_MINUTES_3070: dict[str, float] = {
    "s1_sin_3070": 40.0,
    "s2_rope_3070": 40.0,
    "s3_rope_concat_3070": 40.0,
    "main_ext_3070": 480.0,
}
PROFILES: dict[str, dict[str, float]] = {"colab": BUDGETS_MINUTES, "3070": BUDGETS_MINUTES_3070}


@dataclass
class PlanResult:
    n_steps_measured: int
    warmup_steps_excluded: int
    median_tok_per_sec: float
    seconds_per_optimizer_step: float
    peak_gpu_mem_mb: float
    tokens_per_step: int
    safety_margin: float
    planned_steps: dict[str, int]
    seconds_per_step_basis: str = ""


def load_step_rows(metrics_path: str | Path) -> list[dict]:
    """Rows written by `nmt.train`'s per-step logging: every JSON line with a `tok_per_sec` key
    and no `eval` key is one optimizer step (eval rows nest their payload under `"eval"` and are
    skipped here, as are the one-off `eval_disabled_reason` rows some runs write).
    """
    rows = []
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "tok_per_sec" in row and "eval" not in row:
            rows.append(row)
    return rows


def compute_plan(
    rows: list[dict],
    *,
    tokens_per_step: int = DEFAULT_TOKENS_PER_STEP,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    budgets_minutes: dict[str, float] | None = None,
) -> PlanResult:
    """Steady-state median tokens/s over `rows[warmup_steps:]`, converted to seconds/optimizer-step
    via the *target* `tokens_per_step` (the same knob every later config shares, spec §6) -- not
    each row's own token count, which metrics.jsonl does not log directly. Then
    `planned_steps[label] = floor(budget_minutes * 60 * safety_margin / seconds_per_step)` for each
    entry in `budgets_minutes`.
    """
    budgets_minutes = BUDGETS_MINUTES if budgets_minutes is None else budgets_minutes
    if len(rows) <= warmup_steps:
        raise ValueError(
            f"only {len(rows)} step row(s) in metrics.jsonl; need more than "
            f"warmup_steps={warmup_steps} to measure a steady state"
        )
    steady = rows[warmup_steps:]
    median_tps = statistics.median(r["tok_per_sec"] for r in steady)
    if median_tps <= 0:
        raise ValueError(f"median tok_per_sec is non-positive: {median_tps}")
    if all("wall_step_s" in r for r in steady):
        # Mean whole-iteration wall time: what a wall-clock budget actually pays. tok_per_sec
        # covers compute only (micro-batch assembly runs before its timer starts); on the 3070
        # pilot it implied 0.482 s/step against a measured 0.707 s/step mean wall time, which
        # would have planned "40-minute" ablations that run ~50+ minutes.
        seconds_per_step = statistics.fmean(r["wall_step_s"] for r in steady)
        basis = "mean wall_step_s"
    else:
        seconds_per_step = tokens_per_step / median_tps
        basis = "tokens_per_step / median tok_per_sec (compute-only; older metrics files)"
    peak_mem = max((r.get("gpu_mem_mb", 0.0) for r in rows), default=0.0)

    planned_steps = {
        label: max(1, int((budget_min * 60.0 * safety_margin) // seconds_per_step))
        for label, budget_min in budgets_minutes.items()
    }
    return PlanResult(
        n_steps_measured=len(steady),
        warmup_steps_excluded=warmup_steps,
        median_tok_per_sec=median_tps,
        seconds_per_optimizer_step=seconds_per_step,
        peak_gpu_mem_mb=peak_mem,
        tokens_per_step=tokens_per_step,
        safety_margin=safety_margin,
        planned_steps=planned_steps,
        seconds_per_step_basis=basis,
    )


def write_plan(result: PlanResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive planned_steps for S1/S2/S3/main from a pilot run's metrics.jsonl "
        "(spec §14 P5)."
    )
    parser.add_argument(
        "--metrics", required=True, type=Path, help="Path to the pilot run's metrics.jsonl."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output plan.json path (default: next to --metrics)."
    )
    parser.add_argument("--tokens-per-step", type=int, default=DEFAULT_TOKENS_PER_STEP)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="colab",
        help="Which runs to plan: colab (s1/s2/s3 at 40 min + main at 240) or 3070 "
        "(s1/s2/s3_3070 at 40 min + main_ext_3070 at 480).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = load_step_rows(args.metrics)
    result = compute_plan(
        rows,
        tokens_per_step=args.tokens_per_step,
        warmup_steps=args.warmup_steps,
        safety_margin=args.safety_margin,
        budgets_minutes=PROFILES[args.profile],
    )
    out_path = args.out if args.out is not None else (Path(args.metrics).parent / "plan.json")
    write_plan(result, out_path)

    print(f"wrote {out_path}")
    print(f"steps measured (post-warmup): {result.n_steps_measured}")
    print(f"median tokens/s: {result.median_tok_per_sec:.1f}")
    print(
        f"seconds/optimizer-step: {result.seconds_per_optimizer_step:.3f} "
        f"({result.seconds_per_step_basis})"
    )
    print(f"peak GPU memory: {result.peak_gpu_mem_mb:.1f} MB")
    for label, steps in result.planned_steps.items():
        print(f"PLANNED_STEPS for {label}: {steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
