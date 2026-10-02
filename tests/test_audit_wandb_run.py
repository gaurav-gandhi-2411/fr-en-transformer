from __future__ import annotations

# Pure-logic tests for scripts/audit_wandb_run.py on a small synthetic history, including the
# offline --from-file round trip (no W&B access).
import json
from pathlib import Path
from typing import Any

from scripts import audit_wandb_run as aw


def _history() -> list[dict[str, Any]]:
    rows = []
    vals = {100: 3.0, 200: 2.5, 300: 2.6, 400: 2.7, 500: 2.8}
    for step in range(50, 501, 50):
        r: dict[str, Any] = {
            "step": step,
            "loss": 5.0 - step / 200,
            "grad_norm": 1.0,
            "loss_scale": 1.0,
            "grad_skip_count": 1 if step >= 300 else 0,
            "optimizer_stepped": True,
            "wall_step_s": 0.5,
            "tok_per_sec": 1000.0 + step,
            "epoch_fraction": step / 1000,
        }
        if step in vals:
            r["eval/val_loss"] = vals[step]
            r["eval/e1_bleu"] = step / 10
            r["eval/sample_translations"] = {"_type": "table-file"}
        rows.append(r)
    return rows


META: dict[str, Any] = {
    "info": {"state": "finished"},
    "config": {"git_sha": "abc", "optim": {"planned_steps": 500}, "preflight_gpu": "L4"},
    "summary": {"_runtime": 7200.0, "final_step": 500, "resume_count": 0},
    "metadata": {},
}


def test_rise_runs_requires_two_consecutive_rises() -> None:
    assert aw.rise_runs([3, 2, 2.5, 2.4], [1, 2, 3, 4]) == []
    runs = aw.rise_runs([3, 2, 2.5, 2.6, 2.7, 1], [1, 2, 3, 4, 5, 6])
    assert len(runs) == 1 and runs[0]["n_rises"] == 3 and runs[0]["steps"] == [2, 3, 4, 5]


def test_audit_numbers() -> None:
    a = aw.audit(_history(), META)
    assert a["val_loss_e1"]["min"] == {"step": 200, "val_loss": 2.5}
    assert a["val_loss_e1"]["overfit_ge2_consecutive_rises_after_min"] is True
    assert a["train_loss"]["observed_log_every_steps"] == [50]
    assert a["train_loss"]["loss_at_pct_of_final_step"]["50%"]["step"] == 250
    assert a["stability"]["grad_skip_count"]["increments_between_logged_rows"] == [
        {"step": 300, "from": 0, "to": 1}
    ]
    assert a["stability"]["loss_scale_distinct"] == [1.0]
    assert "eval/sample_translations" not in a["eval_series"]
    assert a["wall"]["cu_used_ESTIMATE"]["cu"] == 2 * aw.CU_PER_HOUR_L4
    assert a["step_anomalies"] == {"step_resets": [], "duplicate_train_steps": []}


def test_nonfinite_and_resume_detection() -> None:
    h = _history()
    h[2]["loss"] = float("nan")
    h.append({**h[0], "step": 50})
    a = aw.audit(h, META)
    assert a["stability"]["nonfinite_loss_rows"] == 1
    assert a["step_anomalies"]["step_resets"][0]["to_step"] == 50
    assert a["step_anomalies"]["duplicate_train_steps"] == [50]


def test_from_file_matches_in_memory(tmp_path: Path) -> None:
    h = _history()
    aw.write_history(tmp_path / "history.jsonl.gz", h)
    (tmp_path / "run_meta.json").write_text(json.dumps(META), encoding="utf-8")
    out = tmp_path / "audit.json"
    assert aw.main(["--from-file", str(tmp_path / "history.jsonl.gz"), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(json.dumps(aw.audit(h, META)))
    assert (tmp_path / "audit.md").is_file()
