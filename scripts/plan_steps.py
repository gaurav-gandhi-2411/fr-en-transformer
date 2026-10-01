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
#   [--safety-margin 0.85]`
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
    seconds_per_step = tokens_per_step / median_tps
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = load_step_rows(args.metrics)
    result = compute_plan(
        rows,
        tokens_per_step=args.tokens_per_step,
        warmup_steps=args.warmup_steps,
        safety_margin=args.safety_margin,
    )
    out_path = args.out if args.out is not None else (Path(args.metrics).parent / "plan.json")
    write_plan(result, out_path)

    print(f"wrote {out_path}")
    print(f"steps measured (post-warmup): {result.n_steps_measured}")
    print(f"median tokens/s: {result.median_tok_per_sec:.1f}")
    print(f"seconds/optimizer-step: {result.seconds_per_optimizer_step:.3f}")
    print(f"peak GPU memory: {result.peak_gpu_mem_mb:.1f} MB")
    for label, steps in result.planned_steps.items():
        print(f"PLANNED_STEPS for {label}: {steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
