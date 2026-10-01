from __future__ import annotations

# Proof that the RTX 3070 queue's interruption handling resumes EXACTLY (run before trusting it on
# the real ablations). It drives the SAME runner the real queue uses (scripts/ablation_3070.py
# `run_training`) with `gpu_pids_fn` injected, against configs/smoke.yaml on synthetic data:
#
#   oom         CPU: --debug-raise-oom-at-step 37 with a checkpoint every 10 steps. Expected: the
#               trainer prints OOM_ABORT step=37 last_ckpt_step=30 and exits 75, the runner resumes
#               from step 30 and finishes at 60; the de-duplicated loss trajectory matches an
#               uninterrupted 60-step run within 1e-4 at every step; redo_steps == 7,
#               resume_count == 1.
#   contention  CPU: the injected GPU view reports a foreign pid mid-run => the runner writes
#               STOP_REQUESTED => the trainer checkpoints at that step boundary and exits 75 => the
#               runner waits until the injected view is free again => resumes; same trajectory
#               check; wait_seconds_total > 0, redo_steps == 0.
#   cuda        tiny CUDA run (envs/cuda), OOM injected at a small step: exit 75 -> resume ->
#               completes, loss finite. CUDA efficient-attention backward is nondeterministic, so
#               exact trajectory equality is only claimed for the CPU scenarios. The injected GPU
#               view is "always free" because this scenario deliberately runs next to another
#               workload; it only needs a few hundred MB.
#
# CLI: `uv run python -m scripts.simulate_resume [--scenarios oom contention cuda]
# [--out reports/resume_simulation] [--work-dir DIR]`. Writes summary.json plus per-scenario log
# excerpts and contention.jsonl under --out. Exit code 1 if any check fails.
import argparse
import json
import math
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from scripts.ablation_3070 import CUDA, REPO, STOP_FILE_NAME, Policy, run_training

CONFIG = "configs/smoke.yaml"
STEPS = 60
CKPT_EVERY = 10
OOM_STEP = 37
FAKE_FOREIGN_PID = 999_999  # never a real pid of ours; stands in for the other workload
KEY_LINES = (
    "CKPT_SAVED",
    "OOM_ABORT",
    "STOPPED_ON_REQUEST",
    "RESUMED",
    "RESUME_CONTEXT",
    "POST_RESUME",
)


def _common_args(run_dir: Path, steps: int, ckpt_every: int) -> list[str]:
    return [
        "--synthetic",
        "--wandb",
        "disabled",
        "--planned-steps",
        str(steps),
        "--ckpt-steps",
        str(ckpt_every),
        "--run-dir",
        str(run_dir),
        "--ckpt-dir",
        str(run_dir / "ckpt"),
    ]


def trajectory(metrics_path: Path) -> dict[int, float]:
    """step -> loss, de-duplicated with the LAST occurrence winning (a redone step is logged twice
    after an OOM abort; the second one is the one the final model was trained with)."""
    out: dict[int, float] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else {}
        if "step" in row and "loss" in row and "eval" not in row:
            out[row["step"]] = row["loss"]
    return out


def _events(contention: Path) -> list[dict]:
    return [json.loads(x) for x in contention.read_text(encoding="utf-8").splitlines() if x]


def _key_lines(train_log: Path) -> list[str]:
    lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines()
    return [x for x in lines if x.startswith(KEY_LINES)]


def run_reference(work: Path, steps: int = STEPS, ckpt_every: int = CKPT_EVERY) -> Path:
    """Uninterrupted CPU run with the same arguments (minus any fault injection)."""
    run_dir = work / "reference"
    cmd = [sys.executable, "-m", "nmt.train", "--config", CONFIG, "--device", "cpu"]
    cmd += _common_args(run_dir, steps, ckpt_every)
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    return run_dir


def _result(
    name: str, run_dir: Path, ref_dir: Path | None, runner_summary: dict, checks: dict[str, bool]
) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    losses = trajectory(run_dir / "metrics.jsonl")
    result: dict = {
        "scenario": name,
        "runner": runner_summary,
        "run_info": {
            k: info.get(k)
            for k in (
                "resumed",
                "resume_count",
                "wait_seconds_total",
                "train_wall_seconds",
                "redo_steps",
                "final_step",
                "exit_reason",
                "device",
                "precision",
            )
        },
        "exit_codes": [
            e["exit_code"] for e in _events(run_dir / "contention.jsonl") if "exit_code" in e
        ],
        "events": [e["event"] for e in _events(run_dir / "contention.jsonl")],
        "key_lines": _key_lines(run_dir / "train_log.txt"),
        "n_steps": len(losses),
        "all_losses_finite": all(math.isfinite(v) for v in losses.values()),
    }
    if ref_dir is not None:
        ref = trajectory(ref_dir / "metrics.jsonl")
        result["trajectory_steps_match"] = set(ref) == set(losses)
        result["max_abs_loss_diff_vs_uninterrupted"] = max(
            abs(ref[s] - losses[s]) for s in set(ref) & set(losses)
        )
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


def _fast_policy() -> Policy:
    return Policy(poll_s=0.1, free_checks=2)


def _own_tree_only(root: int) -> set[int]:
    return {root}  # no powershell process-tree walk in the simulation: the fake pid is foreign


def scenario_oom(work: Path, ref_dir: Path | None = None) -> dict:
    """Forced OOM at step 37 (CPU): exit 75 -> resume from step 30 -> identical trajectory."""
    ref_dir = ref_dir or run_reference(work)
    run_dir = work / "oom"
    args = [*_common_args(run_dir, STEPS, CKPT_EVERY), "--device", "cpu"]
    args += ["--debug-raise-oom-at-step", str(OOM_STEP)]
    summary = run_training(
        CONFIG,
        run_dir,
        args,
        _fast_policy(),
        gpu_pids_fn=lambda: set(),
        cmd_prefix=[sys.executable],
        descendants_fn=_own_tree_only,
    )
    log = (run_dir / "train_log.txt").read_text(encoding="utf-8", errors="replace")
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    ref, got = trajectory(ref_dir / "metrics.jsonl"), trajectory(run_dir / "metrics.jsonl")
    checks = {
        "exit_codes_75_then_0": [
            e["exit_code"] for e in _events(run_dir / "contention.jsonl") if "exit_code" in e
        ]
        == [75, 0],
        "oom_abort_line": f"OOM_ABORT step={OOM_STEP} last_ckpt_step=30" in log,
        "resumed_from_30_exact_state": "RESUMED step=30 " in log and "matches_saved=True" in log,
        "finished_at_planned_steps": info["final_step"] == STEPS,
        "trajectory_matches_1e-4": set(ref) == set(got) == set(range(1, STEPS + 1))
        and all(abs(ref[s] - got[s]) < 1e-4 for s in ref),
        "redo_steps_7": info["redo_steps"] == OOM_STEP - 30,
        "resume_count_1": info["resume_count"] == 1,
        "resumed_true": info["resumed"] is True,
    }
    return _result("oom", run_dir, ref_dir, summary, checks)


class ScriptedForeignGpu:
    """Injected `gpu_pids_fn`: free until the trainer has logged `trigger_rows` steps, then a
    foreign pid is visible until `free_polls` polls have happened AFTER the runner logged the
    stopped trainer's exit 75 (so the runner has to genuinely wait), then free for good."""

    def __init__(self, run_dir: Path, trigger_rows: int = 25, free_polls: int = 6) -> None:
        self.metrics = run_dir / "metrics.jsonl"
        self.stop_file = run_dir / STOP_FILE_NAME
        self.trigger_rows = trigger_rows
        self.free_polls = free_polls
        self.phase = "idle"  # idle -> foreign -> done (one-way: it must not re-trigger on resume)
        self.contention = run_dir / "contention.jsonl"
        self.polls_after_stop = 0
        self._lock = threading.Lock()

    def _rows(self) -> int:
        if not self.metrics.is_file():
            return 0
        return sum(1 for x in self.metrics.read_text(encoding="utf-8").splitlines() if x)

    def _trainer_exited_75(self) -> bool:
        """The runner has logged the stopped trainer's exit, so it is now in its wait loop."""
        if not self.contention.is_file():
            return False
        return '"exit_code": 75' in self.contention.read_text(encoding="utf-8")

    def __call__(self) -> set[int]:
        with self._lock:
            if self.phase == "idle" and self._rows() >= self.trigger_rows:
                self.phase = "foreign"
            if self.phase == "foreign" and self._trainer_exited_75():
                self.polls_after_stop += 1  # counted only while the runner itself is waiting
                if self.polls_after_stop > self.free_polls:
                    self.phase = "done"
            return {FAKE_FOREIGN_PID} if self.phase == "foreign" else set()


def scenario_contention(work: Path, ref_dir: Path | None = None) -> dict:
    """Foreign GPU pid mid-run (CPU): STOP_REQUESTED -> checkpoint, exit 75 -> wait -> resume."""
    ref_dir = ref_dir or run_reference(work)
    run_dir = work / "contention"
    gpu = ScriptedForeignGpu(run_dir)
    summary = run_training(
        CONFIG,
        run_dir,
        [*_common_args(run_dir, STEPS, CKPT_EVERY), "--device", "cpu"],
        _fast_policy(),
        gpu_pids_fn=gpu,
        cmd_prefix=[sys.executable],
        descendants_fn=_own_tree_only,
    )
    log = (run_dir / "train_log.txt").read_text(encoding="utf-8", errors="replace")
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    events = _events(run_dir / "contention.jsonl")
    ref, got = trajectory(ref_dir / "metrics.jsonl"), trajectory(run_dir / "metrics.jsonl")
    stopped = [x for x in log.splitlines() if x.startswith("STOPPED_ON_REQUEST")]
    checks = {
        "exit_codes_75_then_0": [e["exit_code"] for e in events if "exit_code" in e] == [75, 0],
        "stop_requested_logged": any(e["event"] == "stop_requested" for e in events),
        "trainer_stopped_on_request": len(stopped) == 1
        and f"foreign GPU pids [{FAKE_FOREIGN_PID}]" in stopped[0],
        "stop_file_consumed": not (run_dir / STOP_FILE_NAME).exists(),
        "resumed_exact_state": "matches_saved=True" in log,
        "waited_before_resume": any(e["event"] == "wait_end" and e["seconds"] > 0 for e in events),
        "wait_seconds_recorded": info["wait_seconds_total"] > 0,
        "finished_at_planned_steps": info["final_step"] == STEPS,
        "trajectory_matches_1e-4": set(ref) == set(got) == set(range(1, STEPS + 1))
        and all(abs(ref[s] - got[s]) < 1e-4 for s in ref),
        "redo_steps_0": info["redo_steps"] == 0,  # a stop checkpoints at the boundary
        "resume_count_1": info["resume_count"] == 1,
    }
    return _result("contention", run_dir, ref_dir, summary, checks)


def scenario_cuda(work: Path) -> dict:
    """Tiny CUDA run (smoke model) with an injected OOM: exit 75 -> resume -> completes."""
    steps, every, oom_at = 40, 10, 17
    run_dir = work / "cuda"
    args = [*_common_args(run_dir, steps, every), "--device", "cuda"]
    args += ["--debug-raise-oom-at-step", str(oom_at)]
    summary = run_training(
        CONFIG,
        run_dir,
        args,
        _fast_policy(),
        gpu_pids_fn=lambda: set(),
        cmd_prefix=CUDA,
        descendants_fn=_own_tree_only,
    )
    log = (run_dir / "train_log.txt").read_text(encoding="utf-8", errors="replace")
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    losses = trajectory(run_dir / "metrics.jsonl")
    checks = {
        "exit_codes_75_then_0": [
            e["exit_code"] for e in _events(run_dir / "contention.jsonl") if "exit_code" in e
        ]
        == [75, 0],
        "oom_abort_line": f"OOM_ABORT step={oom_at} last_ckpt_step=10" in log,
        "resumed_from_10_exact_state": "RESUMED step=10 " in log and "matches_saved=True" in log,
        "ckpt_saved_lines": "CKPT_SAVED step=40 " in log,
        "ran_on_cuda": "device: cuda" in log,
        "finished_at_planned_steps": info["final_step"] == steps and len(losses) == steps,
        "loss_finite": all(math.isfinite(v) for v in losses.values()),
        "redo_steps": info["redo_steps"] == oom_at - 10,
    }
    return _result("cuda", run_dir, None, summary, checks)


SCENARIOS: dict[str, Callable[..., dict]] = {
    "oom": scenario_oom,
    "contention": scenario_contention,
    "cuda": scenario_cuda,
}


def _scrub(text: str, work: Path) -> str:
    """Replace machine-specific absolute paths (the scratch dir, the repo) with placeholders."""
    for path, tag in ((work, "<work>"), (REPO, "<repo>")):
        text = text.replace(json.dumps(str(path))[1:-1], tag).replace(str(path), tag)
    return text


def _save(out: Path, result: dict, work: Path) -> None:
    name = result["scenario"]
    out.mkdir(parents=True, exist_ok=True)
    run_dir = work / name
    contention = (run_dir / "contention.jsonl").read_text(encoding="utf-8")
    (out / f"{name}_contention.jsonl").write_text(_scrub(contention, work), encoding="utf-8")
    excerpt = "\n".join(result["key_lines"]) + "\n"
    (out / f"{name}_log_excerpt.txt").write_text(_scrub(excerpt, work), encoding="utf-8")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8")) if summary_path.is_file() else {}
    summary[name] = result
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(_scrub(text, work), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume/contention simulation (see preamble).")
    parser.add_argument(
        "--scenarios", nargs="+", choices=sorted(SCENARIOS), default=["oom", "contention"]
    )
    parser.add_argument("--out", default="reports/resume_simulation")
    parser.add_argument("--work-dir", default=None, help="Scratch dir for runs (default: temp).")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="resume_sim_"))
    ref_dir = run_reference(work) if {"oom", "contention"} & set(args.scenarios) else None
    ok = True
    for name in args.scenarios:
        fn = SCENARIOS[name]
        result = fn(work, ref_dir) if name != "cuda" else fn(work)
        _save(out, result, work)
        ok &= result["passed"]
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'} {json.dumps(result['checks'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
