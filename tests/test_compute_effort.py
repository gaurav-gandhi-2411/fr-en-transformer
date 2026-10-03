from __future__ import annotations

import pytest

from scripts import compute_effort as ce


def test_span_hours_accepts_z_and_offsets() -> None:
    assert ce.span_hours("2026-10-01T06:47:16Z", "2026-10-01T08:47:16Z") == pytest.approx(2.0)
    # +05:30 offset: 13:02 IST is 07:32 UTC, so the span to 09:32 UTC is 2 h.
    assert ce.span_hours("2026-09-30T13:02:00+05:30", "2026-09-30T09:32:00Z") == pytest.approx(2.0)


def test_span_hours_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="precedes"):
        ce.span_hours("2026-10-02T00:00:00Z", "2026-10-01T00:00:00Z")


def _run(name: str, group: str, seconds: float | None, rid: str = "x") -> dict[str, object]:
    return {
        "id": rid,
        "name": name,
        "group": group,
        "train_wall_seconds": seconds,
        "wandb_runtime_seconds": 7.0,
    }


def test_compute_gpu_totals_splits_hardware_and_never_guesses_missing_runs() -> None:
    runs = [
        _run("main", "main", 3600.0, "a"),
        _run("ext", "extend_l4", 1800.0, "b"),
        _run("pilot_3070", "pilot_3070", None, "c"),
        _run("s1_sin_3070", "ablation_3070", 36.0, "d"),
    ]
    out = ce.compute_gpu_totals(runs, 2.0)
    assert out["colab_l4"]["hours"] == pytest.approx(1.5)
    assert out["colab_l4"]["cu_ESTIMATE"] == pytest.approx(3.0)
    assert out["local_3070"]["train_wall_seconds"] == pytest.approx(36.0)
    assert out["local_3070"]["runs_without_train_wall_seconds"] == ["pilot_3070"]
    assert out["local_3070"]["wandb_runtime_seconds_of_those"] == {"c": 7.0}
    assert "cu_ESTIMATE" not in out["local_3070"]


def test_crosscheck_reports_per_run_absolute_difference() -> None:
    runs = [_run("a", "main", 10.0, "r1"), _run("b", "main", 20.0, "r2")]
    assert ce.crosscheck(runs, {"r1": 10.5, "r2": 20.0, "missing": 1.0}) == {"r1": 0.5, "r2": 0.0}
