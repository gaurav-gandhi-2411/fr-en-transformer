from __future__ import annotations

# Retry/contention policy of scripts/ablation_3070.py (`run_training`, `wait_until_gpu_free`,
# `run_gpu_stage`), exercised with a fake trainer process (no torch, no GPU) and an injected GPU
# view: exit 75 => wait + resume with the right flags; bounded by MAX_RESUMES / MAX_WAIT_HOURS;
# any other failure is not retried; a completed run is never restarted; a foreign GPU process
# makes the driver write STOP_REQUESTED for a training stage but never interrupts other stages.
import json
import sys
import textwrap
from pathlib import Path

import pytest

from scripts import ablation_3070 as drv
from scripts.ablation_3070 import Policy

# Stand-in for `python -m nmt.train`: run as `python fake.py -m nmt.train --config ... --plan P`.
# Each launch pops the next exit code from the plan, logs its argv, and on exit 0 writes the final
# checkpoint file. "wait_stop" blocks until STOP_REQUESTED appears, consumes it, exits 75.
FAKE_TRAINER = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path
    a = sys.argv
    val = lambda f: a[a.index(f) + 1]
    plan = Path(val("--plan"))
    steps = json.loads(plan.read_text())
    code = steps.pop(0)
    plan.write_text(json.dumps(steps))
    with Path(val("--calls")).open("a") as f:
        f.write(json.dumps(a[a.index("--resume"):]) + "\\n")
    run_dir = Path(val("--run-dir"))
    if code == "wait_stop":
        for _ in range(200):
            if (run_dir / "STOP_REQUESTED").exists():
                (run_dir / "STOP_REQUESTED").unlink()
                sys.exit(75)
            time.sleep(0.05)
        sys.exit(99)
    if code == "short":
        sys.exit(0)
    if code == 0:
        ckpt = Path(val("--ckpt-dir"))
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / f"step_{int(val('--planned-steps')):08d}.pt").write_bytes(b"x")
    sys.exit(code)
    """
)
PLANNED = 10


class Env:
    def __init__(self, tmp: Path, plan: list) -> None:
        self.run_dir, self.ckpt = tmp / "run", tmp / "ckpt"
        self.calls = tmp / "calls.jsonl"
        self.plan = tmp / "plan.json"
        self.plan.write_text(json.dumps(plan))
        (tmp / "fake.py").write_text(FAKE_TRAINER, encoding="utf-8")
        self.prefix = [sys.executable, str(tmp / "fake.py")]
        self.args = [
            "--plan", str(self.plan), "--calls", str(self.calls), "--run-dir", str(self.run_dir),
            "--ckpt-dir", str(self.ckpt), "--planned-steps", str(PLANNED),
        ]  # fmt: skip

    def run(self, policy: Policy, gpu=lambda: set(), **kw):  # noqa: ANN001, ANN003, ANN201
        return drv.run_training(
            "configs/smoke.yaml",
            self.run_dir,
            self.args,
            policy,
            gpu_pids_fn=gpu,
            cmd_prefix=self.prefix,
            descendants_fn=lambda root: {root},
            **kw,
        )

    def launches(self) -> list[list[str]]:
        if not self.calls.is_file():
            return []
        return [json.loads(x) for x in self.calls.read_text().splitlines()]

    def events(self) -> list[dict]:
        text = (self.run_dir / "contention.jsonl").read_text()
        return [json.loads(x) for x in text.splitlines()]


class FakeTime:
    """Virtual clock: `sleep` advances it, so waits cost no real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, s: float) -> None:
        self.now += s

    def clock(self) -> float:
        return self.now


FAST = Policy(poll_s=0.05, free_checks=2)


def test_exit_75_waits_then_resumes_with_flags_and_accumulated_wait(tmp_path: Path) -> None:
    env, ft = Env(tmp_path, [75, 75, 0]), FakeTime()
    seen = {"exits": 0, "busy": 0}

    def gpu() -> set[int]:
        # Free until the first exit 75; after each logged exit the foreign process shows for 3
        # polls (the runner's wait loop; the monitor thread is already stopped by then).
        log = env.run_dir / "contention.jsonl"
        exits = log.read_text().count('"exit_code"') if log.is_file() else 0
        if exits != seen["exits"]:
            seen["exits"], seen["busy"] = exits, 3
        if seen["busy"] > 0:
            seen["busy"] -= 1
            return {4242}
        return set()

    out = env.run(Policy(poll_s=0.05, free_checks=2), gpu, sleep=ft.sleep, clock=ft.clock)
    assert out["status"] == "completed" and out["resumes"] == 2
    launches = env.launches()
    assert [c[c.index("--resume-count") + 1] for c in launches] == ["0", "1", "2"]
    assert all("--resume" in c for c in launches)
    waits = [float(c[c.index("--wait-seconds") + 1]) for c in launches]
    assert waits[0] == 0.0 and 0 < waits[1] < waits[2]  # cumulative, from the virtual clock
    events = [e["event"] for e in env.events()]
    assert events.count("start") == 3 and events.count("resume") == 2
    assert events.count("wait_start") == 2 and events.count("wait_end") == 2
    assert [e["exit_code"] for e in env.events() if "exit_code" in e] == [75, 75, 0]


def test_other_nonzero_exit_fails_immediately_without_retry(tmp_path: Path) -> None:
    env = Env(tmp_path, [3, 0])
    with pytest.raises(SystemExit, match=r"exit 3, not 75"):
        env.run(FAST)
    assert len(env.launches()) == 1


def test_exit_zero_short_of_planned_steps_fails_closed(tmp_path: Path) -> None:
    env = Env(tmp_path, ["short"])  # exit 0 but no final checkpoint (e.g. the wall-clock cap)
    with pytest.raises(SystemExit, match="exited 0 at step 0"):
        env.run(FAST)


def test_max_resumes_fails_closed_after_the_bound(tmp_path: Path) -> None:
    env = Env(tmp_path, [75, 75, 75, 0])
    with pytest.raises(SystemExit, match=r"MAX_RESUMES"):
        env.run(Policy(poll_s=0.05, free_checks=1, max_resumes=2))
    assert len(env.launches()) == 3  # initial + 2 resumes, then fail closed (no 4th launch)
    assert env.events()[-1]["limit"] == "max_resumes"


def test_resume_budget_survives_a_driver_restart(tmp_path: Path) -> None:
    env = Env(tmp_path, [75, 0])
    env.run_dir.mkdir(parents=True)
    rows = [{"event": "resume", "resume": i + 1} for i in range(drv.MAX_RESUMES)]
    (env.run_dir / "contention.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="MAX_RESUMES"):
        env.run(Policy(poll_s=0.05, free_checks=1))  # default max_resumes == 8, already used
    assert len(env.launches()) == 1


def test_max_wait_hours_fails_closed_without_launching(tmp_path: Path) -> None:
    env, ft = Env(tmp_path, [0]), FakeTime()
    policy = Policy(poll_s=3600, free_checks=2, max_wait_hours=2)
    with pytest.raises(SystemExit, match="MAX_WAIT_HOURS"):
        env.run(policy, lambda: {4242}, sleep=ft.sleep, clock=ft.clock)
    assert env.launches() == [] and ft.now > 2 * 3600
    assert env.events()[-1]["limit"] == "max_wait"


def test_completed_run_is_never_restarted(tmp_path: Path) -> None:
    env = Env(tmp_path, [0])
    env.ckpt.mkdir(parents=True)
    (env.ckpt / f"step_{PLANNED:08d}.pt").write_bytes(b"x")
    out = env.run(FAST, lambda: {4242})  # even a busy GPU is irrelevant: nothing to run
    assert out["status"] == "skipped_complete" and env.launches() == []
    assert env.events()[-1]["event"] == "skip_complete"


def test_foreign_process_during_training_writes_stop_request_then_resumes(tmp_path: Path) -> None:
    env = Env(tmp_path, ["wait_stop", 0])
    state = {"foreign": False}

    def gpu() -> set[int]:
        if len(env.launches()) == 1 and not state["foreign"]:
            state["foreign"] = True
            return {4242}
        return set()

    out = env.run(FAST, gpu)
    assert out["status"] == "completed" and out["resumes"] == 1
    events = env.events()
    stop = next(e for e in events if e["event"] == "stop_requested")
    assert "4242" in stop["reason"]
    foreign = next(e for e in events if e["event"] == "foreign")
    assert foreign["action"] == "stop_requested" and foreign["pids"] == [4242]
    assert not (env.run_dir / "STOP_REQUESTED").exists()
    assert [e["exit_code"] for e in events if "exit_code" in e] == [75, 0]


def test_stale_stop_request_is_removed_before_launch(tmp_path: Path) -> None:
    env = Env(tmp_path, [0])
    env.run_dir.mkdir(parents=True)
    (env.run_dir / "STOP_REQUESTED").write_text("stale")
    env.run(FAST)
    assert not (env.run_dir / "STOP_REQUESTED").exists()


def test_non_checkpointable_stage_only_logs_foreign_process(tmp_path: Path) -> None:
    contention = tmp_path / "c.jsonl"
    state = {"polls": 0}

    def gpu() -> set[int]:
        state["polls"] += 1
        return {4242} if state["polls"] > 2 else set()  # free to start, foreign afterwards

    drv.run_gpu_stage(
        [sys.executable, "-c", "import time; time.sleep(0.6)"],
        tmp_path / "log.txt",
        contention,
        None,
        FAST,
        gpu_pids_fn=gpu,
        descendants_fn=lambda root: {root},
    )
    events = [json.loads(x) for x in contention.read_text().splitlines()]
    foreign = [e for e in events if e["event"] == "foreign"]
    assert foreign and all(e["action"] == "none_not_checkpointable" for e in foreign)
    assert not any(e["event"] == "stop_requested" for e in events)
    assert events[-1] == {**events[-1], "event": "end", "exit_code": 0}


def test_wait_until_gpu_free_reports_zero_when_never_busy(tmp_path: Path) -> None:
    ft = FakeTime()
    waited = drv.wait_until_gpu_free(
        tmp_path / "c.jsonl", Policy(poll_s=60), lambda: set(), None, ft.sleep, ft.clock
    )
    assert waited == 0.0 and not (tmp_path / "c.jsonl").exists()
