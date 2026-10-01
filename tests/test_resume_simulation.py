from __future__ import annotations

# End-to-end proof (CPU, configs/smoke.yaml on synthetic data, real `python -m nmt.train`
# subprocesses driven by the real scripts/ablation_3070.py runner with an injected GPU view) that
# an OOM abort and a contention stop both resume EXACTLY: same 60-step loss trajectory as an
# uninterrupted run within 1e-4. The CUDA scenario is run by hand (needs the GPU); CUDA
# efficient-attention backward is nondeterministic, so exact equality is claimed on CPU only.
import json
from pathlib import Path

import pytest

from scripts import simulate_resume as sim


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    work = tmp_path_factory.mktemp("resume_sim")
    return work, sim.run_reference(work)


def test_oom_at_37_resumes_from_30_and_matches_uninterrupted(reference: tuple[Path, Path]) -> None:
    work, ref_dir = reference
    result = sim.scenario_oom(work, ref_dir)
    assert result["passed"], result["checks"]
    assert result["run_info"]["redo_steps"] == 7 and result["run_info"]["resume_count"] == 1
    assert result["exit_codes"] == [75, 0]
    assert result["max_abs_loss_diff_vs_uninterrupted"] < 1e-4
    assert any(x.startswith("OOM_ABORT step=37 last_ckpt_step=30") for x in result["key_lines"])


def test_foreign_gpu_process_stops_waits_and_resumes_exactly(reference: tuple[Path, Path]) -> None:
    work, ref_dir = reference
    result = sim.scenario_contention(work, ref_dir)
    assert result["passed"], result["checks"]
    assert result["run_info"]["wait_seconds_total"] > 0 and result["run_info"]["redo_steps"] == 0
    assert result["max_abs_loss_diff_vs_uninterrupted"] < 1e-4
    assert "stop_requested" in result["events"] and "wait_end" in result["events"]


def test_trajectory_dedupes_redone_steps_keeping_the_last_occurrence(tmp_path: Path) -> None:
    rows = [
        {"step": 1, "loss": 5.0},
        {"step": 2, "loss": 4.0},
        {"step": 2, "loss": 3.5},  # redone after an OOM abort
        {"eval": {"step": 2, "val_loss": 9.0}},
        {"eval_disabled_reason": "x"},
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert sim.trajectory(path) == {1: 5.0, 2: 3.5}
