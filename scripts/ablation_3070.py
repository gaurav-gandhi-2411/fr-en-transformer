from __future__ import annotations

# Local RTX 3070 ablation driver (PREREG.md §3). Runs S1/S2/S3 sequentially with identical
# data/seed/planned_steps, then for each final checkpoint: export -> full E1+E2 decoding tuning
# (nmt.tune) -> evaluation at the tuned alpha/beam with segmentation OFF (primary for H1/H2) and at
# the tuned T (secondary), COMET on -> the pre-registered paired comparisons (nmt.compare).
#
# The GPU is shared with other workloads on this laptop: before each stage it waits until no
# foreign process holds the GPU, and while a stage runs it records every foreign compute process
# it sees (runs/<cfg>/contention.jsonl). A foreign process on an 8 GB card can push WDDM into
# spilling to system RAM -- that happened once (pilot attempt 2) -- so contention is evidence, never
# silently ignored. Every stage is idempotent, so the driver can simply be re-run after any
# interruption.
#
# Training stage (run_training) is contention- and OOM-aware, with exact resumption:
#   * a foreign GPU process during training => write <run_dir>/STOP_REQUESTED; the trainer
#     checkpoints at that step boundary and exits 75 (EX_TEMPFAIL);
#   * exit 75 (that stop, or an OOM abort that rolls back to the last periodic checkpoint) =>
#     wait for a free GPU, relaunch with --resume --resume-count K --wait-seconds W. Because the
#     checkpoint holds model/optimizer/sampler/RNG state, data order and step count equal an
#     uninterrupted run (PREREG §3 fairness: identical planned_steps, seed, data order);
#   * bounded: MAX_RESUMES per run and MAX_WAIT_HOURS of waiting per run, then fail closed; any
#     other non-zero exit fails immediately (no blind retry); a run already at planned_steps is
#     never restarted.
# Non-checkpointable stages (export/tune/eval/COMET) are never interrupted: contention is logged
# and the stage finishes (deterministic, re-runnable outputs).
#
# CLI: `python scripts/ablation_3070.py [--stages train eval compare] [--configs ...]`
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = ("s1_sin_3070", "s2_rope_3070", "s3_rope_concat_3070")
CUDA = ["uv", "run", "--project", "envs/cuda", "python"]
POLL_S = 60
FREE_CHECKS = 2  # consecutive foreign-free polls required before starting a GPU stage
EX_TEMPFAIL = 75  # nmt.train exit code for a contention stop / OOM abort: resumable
MAX_RESUMES = 8  # per run; the next interruption fails closed instead of retrying forever
MAX_WAIT_HOURS = 24  # per run: total time spent waiting for the GPU, then fail closed
STOP_FILE_NAME = "STOP_REQUESTED"  # == nmt.train.STOP_FILE_NAME (literal: this driver avoids torch)
TELEMETRY_FIELDS = (
    "timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,"
    "clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,"
    "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown"
)


@dataclass(frozen=True)
class Policy:
    """Retry/contention limits for one training run; exceeding a bound fails closed."""

    max_resumes: int = MAX_RESUMES
    max_wait_hours: float = MAX_WAIT_HOURS
    poll_s: float = POLL_S
    free_checks: int = FREE_CHECKS


DEFAULT_POLICY = Policy()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def gpu_pids() -> set[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {int(x) for x in out.split() if x.strip().isdigit()}


def descendants(root: int) -> set[int]:
    """PIDs of `root` and all its descendants (Windows: via CIM; uv spawns python children)."""
    ps = (
        'Get-CimInstance Win32_Process | ForEach-Object { "$($_.ProcessId) $($_.ParentProcessId)" }'
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=True
    ).stdout
    children: dict[int, list[int]] = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def wait_until_gpu_free(
    log: Path,
    policy: Policy = DEFAULT_POLICY,
    gpu_pids_fn: Callable[[], set[int]] | None = None,
    max_wait_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Block until `policy.free_checks` consecutive polls see no GPU process. Returns the seconds
    spent waiting (first busy poll -> confirmed free; 0.0 if the GPU was never busy) and logs
    wait_start / wait / wait_end events. Fails closed (SystemExit) once the busy wait exceeds
    `max_wait_s`."""
    pids_fn = gpu_pids_fn or gpu_pids
    free_streak = 0
    busy_since: float | None = None
    while free_streak < policy.free_checks:
        pids = pids_fn()
        if pids:
            free_streak = 0
            if busy_since is None:
                busy_since = clock()
                _append(log, {"time": _now(), "event": "wait_start", "foreign_pids": sorted(pids)})
            print(
                f"[{_now()}] GPU busy (pids {sorted(pids)}); waiting {policy.poll_s}s", flush=True
            )
            _append(log, {"time": _now(), "event": "wait", "foreign_pids": sorted(pids)})
            if max_wait_s is not None and clock() - busy_since > max_wait_s:
                _append(log, {"time": _now(), "event": "policy_exceeded", "limit": "max_wait"})
                raise SystemExit(
                    f"GPU still busy after {(clock() - busy_since) / 3600:.2f} h of waiting: the "
                    f"wait budget (MAX_WAIT_HOURS={policy.max_wait_hours}) is exhausted; stopping "
                    "(fail closed). Re-run the driver once the GPU is free."
                )
        else:
            free_streak += 1
        if free_streak < policy.free_checks:
            sleep(policy.poll_s)
    waited = 0.0 if busy_since is None else clock() - busy_since
    if busy_since is not None:
        _append(log, {"time": _now(), "event": "wait_end", "seconds": round(waited, 3)})
    return waited


def _request_stop(stop_file: Path, reason: str) -> None:
    """Atomically create the trainer's STOP_REQUESTED file (it never sees a half-written one)."""
    tmp = stop_file.with_name(f".{stop_file.name}.tmp")
    tmp.write_text(reason, encoding="utf-8")
    os.replace(tmp, stop_file)


def _run_process(
    cmd: list[str],
    log_path: Path,
    contention: Path,
    telemetry: Path | None,
    *,
    stop_file: Path | None,
    policy: Policy,
    gpu_pids_fn: Callable[[], set[int]],
    descendants_fn: Callable[[int], set[int]],
) -> int:
    """Run `cmd` (stdout/stderr appended to log_path); every `policy.poll_s` seconds record
    nvidia-smi telemetry (if `telemetry`) and foreign GPU processes (pids outside the command's own
    process tree). With `stop_file` (a checkpointable training stage) the first foreign process
    makes the driver write it, asking the trainer to checkpoint and exit 75; without it (eval /
    tune / export) the event is only logged. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()

        def sample() -> None:
            if telemetry is not None and not telemetry.exists():
                subprocess.run(
                    ["nvidia-smi", f"--query-gpu={TELEMETRY_FIELDS}", "--format=csv"],
                    stdout=telemetry.open("w", encoding="utf-8"),
                    check=False,
                )
            stop_requested = False
            while not stop.is_set():
                try:
                    if telemetry is not None:
                        with telemetry.open("a", encoding="utf-8") as t:
                            subprocess.run(
                                [
                                    "nvidia-smi",
                                    f"--query-gpu={TELEMETRY_FIELDS}",
                                    "--format=csv,noheader",
                                ],
                                stdout=t,
                                check=False,
                            )
                    pids = gpu_pids_fn()
                    foreign = pids - descendants_fn(proc.pid) if pids else set()
                    if foreign:
                        action = "stop_requested" if stop_file else "none_not_checkpointable"
                        row = {"time": _now(), "event": "foreign", "pids": sorted(foreign)}
                        _append(contention, {**row, "action": action})
                        if stop_file is not None and not stop_requested:
                            reason = f"foreign GPU pids {sorted(foreign)} at {_now()}"
                            _request_stop(stop_file, reason)
                            stop_requested = True
                            _append(
                                contention,
                                {"time": _now(), "event": "stop_requested", "reason": reason},
                            )
                except Exception as exc:  # noqa: BLE001 - a monitor hiccup must not end monitoring
                    _append(
                        contention, {"time": _now(), "event": "monitor_error", "error": repr(exc)}
                    )
                stop.wait(policy.poll_s)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        code = proc.wait()
        stop.set()
        sampler.join(timeout=30)
    return code


def run_gpu_stage(
    cmd: list[str],
    log_path: Path,
    contention: Path,
    telemetry: Path | None,
    policy: Policy = DEFAULT_POLICY,
    gpu_pids_fn: Callable[[], set[int]] | None = None,
    descendants_fn: Callable[[int], set[int]] | None = None,
) -> None:
    """Run a NON-checkpointable `cmd` (export/tune/eval/COMET) once the GPU is free. Foreign GPU
    processes are logged to `contention` but never interrupt it (its outputs are deterministic
    and simply re-run on failure); a non-zero exit fails the driver."""
    pids_fn = gpu_pids_fn or gpu_pids
    wait_until_gpu_free(contention, policy, pids_fn)
    _append(contention, {"time": _now(), "event": "start", "cmd": cmd})
    code = _run_process(
        cmd,
        log_path,
        contention,
        telemetry,
        stop_file=None,
        policy=policy,
        gpu_pids_fn=pids_fn,
        descendants_fn=descendants_fn or descendants,
    )
    _append(contention, {"time": _now(), "event": "end", "exit_code": code})
    if code != 0:
        raise SystemExit(f"stage failed (exit {code}): {' '.join(cmd)} -- see {log_path}")


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    """Value of `--flag X` in an argv list (last occurrence), or None."""
    value = None
    for i, a in enumerate(args[:-1]):
        if a == flag:
            value = args[i + 1]
    return value


def latest_ckpt_step(ckpt_dir: Path) -> int:
    """Highest step among `step_XXXXXXXX.pt` files in `ckpt_dir` (0 if none)."""
    matches = (re.fullmatch(r"step_(\d+)\.pt", f.name) for f in ckpt_dir.glob("step_*.pt"))
    return max((int(m.group(1)) for m in matches if m), default=0)


def _history(contention: Path) -> tuple[int, float]:
    """(resumes so far, cumulative seconds waited) read from the run's contention.jsonl, so
    MAX_RESUMES and MAX_WAIT_HOURS are per RUN even if the driver itself is restarted."""
    resumes, waited = 0, 0.0
    if contention.is_file():
        for line in contention.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            if row.get("event") == "resume":
                resumes += 1
            elif row.get("event") == "wait_end":
                waited += float(row.get("seconds", 0.0))
    return resumes, waited


def run_training(
    config_path: str | Path,
    run_dir: Path,
    extra_args: Sequence[str] = (),
    policy: Policy = DEFAULT_POLICY,
    gpu_pids_fn: Callable[[], set[int]] | None = None,
    *,
    cmd_prefix: Sequence[str] = CUDA,
    descendants_fn: Callable[[int], set[int]] | None = None,
    telemetry: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Train one config to its planned_steps through GPU contention and OOM, resuming exactly.

    Idempotent: a run whose latest checkpoint already reached planned_steps is never restarted.
    Each launch waits until the GPU is free, then runs `python -m nmt.train --config ... --resume
    --resume-count K --wait-seconds W`. While it runs, a foreign GPU process makes the driver
    write `<run_dir>/STOP_REQUESTED` (the trainer checkpoints at that step boundary, exits 75).
    Exit 75 (contention stop or OOM abort) => wait for a free GPU and resume, at most
    `policy.max_resumes` times and `policy.max_wait_hours` of waiting per run; any other non-zero
    exit, or exit 0 short of planned_steps, fails immediately (no blind retry). Every event is
    appended to `<run_dir>/contention.jsonl`. `gpu_pids_fn`/`descendants_fn`/`sleep`/`clock` are
    injectable so the policy can be simulated without a GPU (scripts/simulate_resume.py).
    """
    pids_fn = gpu_pids_fn or gpu_pids
    contention = run_dir / "contention.jsonl"
    stop_file = run_dir / STOP_FILE_NAME
    raw = yaml.safe_load((REPO / config_path).read_text(encoding="utf-8")) or {}
    planned_flag = _flag_value(extra_args, "--planned-steps")
    planned = int(planned_flag) if planned_flag else int(raw["optim"]["planned_steps"])
    ckpt_dir = Path(_flag_value(extra_args, "--ckpt-dir") or raw["ckpt"]["dir"])
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else REPO / ckpt_dir
    resumes, waited_total = _history(contention)
    max_wait_s = policy.max_wait_hours * 3600

    while True:
        done_step = latest_ckpt_step(ckpt_dir)
        if done_step >= planned:
            _append(contention, {"time": _now(), "event": "skip_complete", "step": done_step})
            return {"status": "skipped_complete", "resumes": resumes, "wait_seconds": waited_total}
        waited_total += wait_until_gpu_free(
            contention, policy, pids_fn, max_wait_s - waited_total, sleep, clock
        )
        stop_file.unlink(missing_ok=True)  # a stale request would stop the new launch at step 1
        cmd = [
            *cmd_prefix,
            "-m",
            "nmt.train",
            "--config",
            str(config_path),
            *extra_args,
            "--resume",
            "--resume-count",
            str(resumes),
            "--wait-seconds",
            f"{waited_total:.3f}",
        ]
        start = {"time": _now(), "event": "start", "cmd": cmd, "resume": resumes}
        _append(contention, {**start, "wait_seconds_total": round(waited_total, 3)})
        code = _run_process(
            cmd,
            run_dir / "train_log.txt",
            contention,
            telemetry,
            stop_file=stop_file,
            policy=policy,
            gpu_pids_fn=pids_fn,
            descendants_fn=descendants_fn or descendants,
        )
        _append(contention, {"time": _now(), "event": "end", "exit_code": code})
        if code == 0:
            reached = latest_ckpt_step(ckpt_dir)
            if reached < planned:
                raise SystemExit(
                    f"trainer exited 0 at step {reached} < planned_steps {planned} (wall-clock "
                    f"cap?); PREREG §3 invalidates such a run -- see {run_dir / 'train_log.txt'}"
                )
            return {"status": "completed", "resumes": resumes, "wait_seconds": waited_total}
        if code != EX_TEMPFAIL:
            raise SystemExit(
                f"training failed (exit {code}, not {EX_TEMPFAIL}): {' '.join(cmd)} -- see "
                f"{run_dir / 'train_log.txt'}; not retried"
            )
        if resumes + 1 > policy.max_resumes:
            limit = {"time": _now(), "event": "policy_exceeded", "limit": "max_resumes"}
            _append(contention, limit)
            raise SystemExit(
                f"trainer exited {EX_TEMPFAIL} again after {policy.max_resumes} resumes "
                f"(MAX_RESUMES); stopping (fail closed) -- see {contention}"
            )
        resumes += 1
        _append(contention, {"time": _now(), "event": "resume", "resume": resumes})


def train(cfg: str) -> None:
    run_dir = REPO / "runs" / cfg
    run_training(
        f"configs/{cfg}.yaml",
        run_dir,
        ["--wandb", "online"],
        telemetry=run_dir / "gpu_samples.csv",
    )


def evaluate(cfg: str) -> None:
    report = REPO / "reports" / cfg
    run_dir = REPO / "runs" / cfg
    stage = [*CUDA, "-m", "nmt.pipeline", "--config", f"configs/{cfg}.yaml", "--seed", "1234"]

    def gpu(cmd: list[str], name: str) -> None:
        run_gpu_stage(
            cmd,
            run_dir / f"{name}_log.txt",
            run_dir / "contention.jsonl",
            run_dir / f"gpu_samples_{name}.csv",
        )

    export = report / "export"
    if not (export / "model.safetensors").is_file():
        gpu([*stage, "--stage", "export", "--run-dir", str(report)], "export")
    grid = report / "selection_grid.json"
    if not grid.is_file():
        gpu([*CUDA, "-m", "nmt.tune", "--model", str(export), "--out", str(grid)], "tune")
    winner = json.loads(grid.read_text(encoding="utf-8"))["winner"]
    variants = {
        "primary_segoff": None,  # PREREG §3: primary decoding for H1/H2
        "secondary_tuned": winner["segment_threshold"],
    }
    for name, threshold in variants.items():
        out = report / name
        if (out / "eval" / "export" / "eval.json").is_file():
            continue
        cmd = [
            *stage,
            "--stage",
            "evaluate",
            "--model",
            str(export),
            "--beam",
            str(winner["beam"]),
            "--alpha",
            str(winner["alpha"]),
            "--n-bootstrap",
            "1000",
            "--comet",
            "--run-dir",
            str(out),
        ]
        if threshold is not None:
            cmd += ["--segment-threshold", str(threshold)]
        gpu(cmd, f"eval_{name}")


COMET_TOL = 1e-3  # max |system score| difference GPU vs the committed CPU sanity run


def comet_gpu_parity() -> None:
    """Fail closed unless COMET-22 on the GPU reproduces the committed CPU sanity scores (same 150
    triples): the env switched to the CUDA torch build and its GPU path was never verified."""
    gpu_out = REPO / "reports" / "audit" / "comet_sanity_gpu.json"
    if not gpu_out.is_file():
        run_gpu_stage(
            [
                *CUDA,
                "-m",
                "scripts.comet_sanity",
                "--pred",
                "reports/smoke_tuned/eval/export/dev_predictions.json",
                "--out",
                str(gpu_out),
            ],
            REPO / "runs" / "comet_parity" / "log.txt",
            REPO / "runs" / "comet_parity" / "contention.jsonl",
            REPO / "runs" / "comet_parity" / "gpu_samples.csv",
        )
    cpu = json.loads((REPO / "reports" / "audit" / "comet_sanity.json").read_text("utf-8"))
    gpu = json.loads(gpu_out.read_text("utf-8"))
    for name, c in cpu["conditions"].items():
        diff = abs(c["system_score"] - gpu["conditions"][name]["system_score"])
        print(f"COMET parity {name}: cpu={c['system_score']:.4f} diff={diff:.2e}", flush=True)
        if diff > COMET_TOL:
            raise SystemExit(f"COMET GPU/CPU parity failed for {name}: |diff|={diff} > {COMET_TOL}")


def compare() -> None:
    pairs = {
        "H1_s2_vs_s1": ("s2_rope_3070", "s1_sin_3070"),
        "H2_s3_vs_s2": (
            "s3_rope_concat_3070",
            "s2_rope_3070",
        ),
    }
    out_dir = REPO / "reports" / "ablation_3070"
    for label, (a, b) in pairs.items():
        for variant in ("primary_segoff", "secondary_tuned"):
            out = out_dir / f"{label}_{variant}.json"
            subprocess.run(
                [
                    *CUDA,
                    "-m",
                    "nmt.compare",
                    "--a",
                    str(REPO / "reports" / a / variant / "eval" / "export"),
                    "--b",
                    str(REPO / "reports" / b / variant / "eval" / "export"),
                    "--out",
                    str(out),
                    "--objective",
                ],
                cwd=REPO,
                check=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RTX 3070 ablation driver (PREREG.md §3).")
    parser.add_argument("--stages", nargs="+", default=["train", "eval", "compare"])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = parser.parse_args(argv)
    os.chdir(REPO)
    if "train" in args.stages:
        for cfg in args.configs:
            print(f"[{_now()}] train {cfg}", flush=True)
            train(cfg)
    if "eval" in args.stages:
        comet_gpu_parity()
        for cfg in args.configs:
            print(f"[{_now()}] eval {cfg}", flush=True)
            evaluate(cfg)
    if "compare" in args.stages:
        compare()
    print(f"[{_now()}] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
