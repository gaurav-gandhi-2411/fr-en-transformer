from __future__ import annotations

# Unit tests for scripts/plan_steps.py against a synthetic metrics.jsonl (spec §14 P5: "Unit-test
# plan_steps on a synthetic metrics.jsonl").
import json
from pathlib import Path

import pytest

from scripts.plan_steps import (
    DEFAULT_SAFETY_MARGIN,
    DEFAULT_WARMUP_STEPS,
    compute_plan,
    load_step_rows,
    main,
    write_plan,
)


def _write_synthetic_metrics(
    path: Path, n_steps: int, tok_per_sec: float, gpu_mem_mb: float
) -> None:
    """A metrics.jsonl with `n_steps` constant-throughput step rows, an eval row (which must be
    skipped), and a one-off `eval_disabled_reason` row (also skipped)."""
    lines = [json.dumps({"eval_disabled_reason": "e1 split not found: ..."})]
    for step in range(1, n_steps + 1):
        lines.append(
            json.dumps(
                {
                    "step": step,
                    "loss": 5.0,
                    "tok_per_sec": tok_per_sec,
                    "gpu_mem_mb": gpu_mem_mb,
                    "lr": 1e-4,
                }
            )
        )
        if step % 10 == 0:
            lines.append(json.dumps({"eval": {"step": step, "val_loss": 4.5}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_step_rows_skips_eval_and_disabled_reason_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    _write_synthetic_metrics(metrics_path, n_steps=30, tok_per_sec=1000.0, gpu_mem_mb=500.0)
    rows = load_step_rows(metrics_path)
    assert len(rows) == 30
    assert all("eval" not in r and "eval_disabled_reason" not in r for r in rows)
    assert [r["step"] for r in rows] == list(range(1, 31))


def test_compute_plan_constant_throughput_matches_expected_steps(tmp_path: Path) -> None:
    """tok_per_sec=25000/sec, tokens_per_step=25000 -> exactly 1 second/optimizer-step, so
    planned_steps[label] == floor(budget_minutes * 60 * safety_margin)."""
    metrics_path = tmp_path / "metrics.jsonl"
    _write_synthetic_metrics(metrics_path, n_steps=100, tok_per_sec=25000.0, gpu_mem_mb=12000.0)
    rows = load_step_rows(metrics_path)

    result = compute_plan(rows, tokens_per_step=25000, warmup_steps=20, safety_margin=0.85)

    assert result.n_steps_measured == 80  # 100 - 20 warmup
    assert result.warmup_steps_excluded == 20
    assert result.median_tok_per_sec == pytest.approx(25000.0)
    assert result.seconds_per_optimizer_step == pytest.approx(1.0)
    assert result.peak_gpu_mem_mb == pytest.approx(12000.0)
    assert result.planned_steps["s1_sin"] == int(40 * 60 * 0.85)
    assert result.planned_steps["s2_rope"] == int(40 * 60 * 0.85)
    assert result.planned_steps["main"] == int(240 * 60 * 0.85)


def test_compute_plan_excludes_warmup_from_the_measurement(tmp_path: Path) -> None:
    """A slow warmup phase followed by a fast steady state: the recommendation must reflect only
    the steady-state throughput, not be dragged down by the (excluded) warmup rows."""
    metrics_path = tmp_path / "metrics.jsonl"
    lines = []
    for step in range(1, 11):  # slow warmup: 1000 tok/s
        lines.append(json.dumps({"step": step, "tok_per_sec": 1000.0, "gpu_mem_mb": 100.0}))
    for step in range(11, 61):  # steady state: 20000 tok/s
        lines.append(json.dumps({"step": step, "tok_per_sec": 20000.0, "gpu_mem_mb": 8000.0}))
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = load_step_rows(metrics_path)
    result = compute_plan(rows, tokens_per_step=25000, warmup_steps=10)

    assert result.median_tok_per_sec == pytest.approx(20000.0)
    assert result.n_steps_measured == 50


def test_compute_plan_raises_with_too_few_rows() -> None:
    rows = [{"step": i, "tok_per_sec": 1000.0} for i in range(1, 5)]
    with pytest.raises(ValueError, match="steady state"):
        compute_plan(rows, warmup_steps=DEFAULT_WARMUP_STEPS)


def test_write_plan_round_trips_through_json(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    _write_synthetic_metrics(metrics_path, n_steps=50, tok_per_sec=25000.0, gpu_mem_mb=9000.0)
    rows = load_step_rows(metrics_path)
    result = compute_plan(rows)
    out_path = write_plan(result, tmp_path / "plan.json")
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["planned_steps"]["main"] == result.planned_steps["main"]
    assert loaded["safety_margin"] == DEFAULT_SAFETY_MARGIN


def test_main_cli_writes_plan_json_next_to_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    _write_synthetic_metrics(metrics_path, n_steps=50, tok_per_sec=25000.0, gpu_mem_mb=9000.0)

    exit_code = main(["--metrics", str(metrics_path)])

    assert exit_code == 0
    plan_path = tmp_path / "plan.json"
    assert plan_path.is_file()
    captured = capsys.readouterr()
    assert "PLANNED_STEPS for main:" in captured.out


def test_main_cli_3070_profile_plans_the_3070_runs_only(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """1 s/step synthetic pilot: 40-min ablations plan int(40*60*0.85) steps, identical across
    s1/s2/s3_3070; main_ext_3070 plans 2x main's 240-minute budget; colab labels are absent."""
    metrics_path = tmp_path / "metrics.jsonl"
    _write_synthetic_metrics(metrics_path, n_steps=50, tok_per_sec=25000.0, gpu_mem_mb=9000.0)

    assert main(["--metrics", str(metrics_path), "--profile", "3070"]) == 0

    planned = json.loads((tmp_path / "plan.json").read_text())["planned_steps"]
    assert set(planned) == {"s1_sin_3070", "s2_rope_3070", "s3_rope_concat_3070", "main_ext_3070"}
    assert planned["s1_sin_3070"] == planned["s2_rope_3070"] == planned["s3_rope_concat_3070"]
    assert planned["s1_sin_3070"] == int(40 * 60 * 0.85)
    assert planned["main_ext_3070"] == int(480 * 60 * 0.85)
    assert "PLANNED_STEPS for main_ext_3070:" in capsys.readouterr().out
