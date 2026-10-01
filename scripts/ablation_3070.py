from __future__ import annotations

# Local RTX 3070 ablation driver (PREREG.md §3). Runs S1/S2/S3 sequentially with identical
# data/seed/planned_steps, then for each final checkpoint: export -> full E1+E2 decoding tuning
# (nmt.tune) -> evaluation at the tuned alpha/beam with segmentation OFF (primary for H1/H2) and at
# the tuned T (secondary), COMET on -> the pre-registered paired comparisons (nmt.compare).
#
# The GPU is shared with other workloads on this laptop: before each stage it waits until no
# foreign process holds the GPU, and while a stage runs it records every foreign compute process
# it sees (runs/<cfg>/contention.jsonl). A foreign process on an 8 GB card can push WDDM into
# spilling to system RAM -- that happened once (pilot attempt 2) -- so contention is logged as
# evidence, never silently ignored. Every stage is idempotent (training resumes; finished
# outputs are skipped), so the driver can simply be re-run after any interruption.
#
# CLI: `python scripts/ablation_3070.py [--stages train eval compare] [--configs ...]`
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = ("s1_sin_3070", "s2_rope_3070", "s3_rope_concat_3070")
CUDA = ["uv", "run", "--project", "envs/cuda", "python"]
POLL_S = 60
FREE_CHECKS = 2  # consecutive foreign-free polls required before starting a GPU stage


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


def wait_until_gpu_free(log: Path) -> None:
    free_streak = 0
    while free_streak < FREE_CHECKS:
        pids = gpu_pids()
        if pids:
            free_streak = 0
            print(f"[{_now()}] GPU busy (pids {sorted(pids)}); waiting {POLL_S}s", flush=True)
            _append(log, {"time": _now(), "event": "wait", "foreign_pids": sorted(pids)})
        else:
            free_streak += 1
        if free_streak < FREE_CHECKS:
            time.sleep(POLL_S)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_gpu_stage(cmd: list[str], log_path: Path, contention: Path, telemetry: Path) -> None:
    """Run `cmd` (stdout/stderr appended to log_path) after the GPU is free, sampling
    nvidia-smi telemetry and foreign GPU processes every POLL_S seconds while it runs."""
    wait_until_gpu_free(contention)
    _append(contention, {"time": _now(), "event": "start", "cmd": cmd})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()

        def sample() -> None:
            fields = (
                "timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,"
                "clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,"
                "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown"
            )
            if not telemetry.exists():
                subprocess.run(
                    ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
                    stdout=telemetry.open("w", encoding="utf-8"),
                    check=False,
                )
            while not stop.is_set():
                with telemetry.open("a", encoding="utf-8") as t:
                    subprocess.run(
                        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                        stdout=t,
                        check=False,
                    )
                foreign = gpu_pids() - descendants(proc.pid)
                if foreign:
                    _append(
                        contention, {"time": _now(), "event": "foreign", "pids": sorted(foreign)}
                    )
                stop.wait(POLL_S)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        code = proc.wait()
        stop.set()
        sampler.join(timeout=30)
    _append(contention, {"time": _now(), "event": "end", "exit_code": code})
    if code != 0:
        raise SystemExit(f"stage failed (exit {code}): {' '.join(cmd)} -- see {log_path}")


def train(cfg: str) -> None:
    run_dir = REPO / "runs" / cfg
    run_gpu_stage(
        [
            *CUDA,
            "-m",
            "nmt.train",
            "--config",
            f"configs/{cfg}.yaml",
            "--resume",
            "--wandb",
            "online",
        ],
        run_dir / "train_log.txt",
        run_dir / "contention.jsonl",
        run_dir / "gpu_samples.csv",
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
