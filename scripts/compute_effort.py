from __future__ import annotations

# Elapsed span, GPU hours and Colab compute units (CU) for the report's effort statement, written
# to reports/final/effort_compute.json with provenance. Nothing here estimates human hours: the
# "about 8-9 hours of my hands-on time" figure is the owner's own and is not derivable from files.
#
# Sources:
#   git      first and last commit author date on --ref (default origin/main), via `git log`
#   W&B      read-only Api().runs(...): created_at of every run, heartbeat_at (last heartbeat, used
#            as the finished time: W&B exposes no finished_at), summary train_wall_seconds
#   files    reports/final/wandb_run_summaries.json (main, S1, S2, S3) and
#            reports/extension/ext_val_loss_summary.json (extension; read from --ext-ref when the
#            file is not in the working tree because its PR has not merged)
#   rate     reports/pilot_l4/pilot_summary.json colab_pro_compute_units_per_hour (the rate the
#            owner reported); CU = hours x rate is an ESTIMATE, not a Colab ledger figure.
#
# Usage: python -m scripts.compute_effort [--ref origin/main] [--out FILE]
import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTITY_PROJECT = "gauravgandhi429-gaurav-gandhi/fr-en-transformer"
SUMMARIES = "reports/final/wandb_run_summaries.json"
EXT_SUMMARY = "reports/extension/ext_val_loss_summary.json"
PILOT_SUMMARY = "reports/pilot_l4/pilot_summary.json"
# W&B groups that ran on the local RTX 3070 (no Colab CU). Every other group ran on a Colab L4;
# the "pilot" group is the Colab L4 pilot (reports/pilot_l4/pilot_summary.json, gpu NVIDIA L4).
LOCAL_3070_GROUPS = ("pilot_3070", "ablation_3070")


def _iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, accepting a trailing Z."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def span_hours(first: str, last: str) -> float:
    """Hours between two ISO-8601 timestamps (last minus first); raises if last precedes first."""
    delta = (_iso(last) - _iso(first)).total_seconds()
    if delta < 0:
        raise ValueError(f"last ({last}) precedes first ({first})")
    return delta / 3600.0


def compute_gpu_totals(runs: list[dict[str, Any]], cu_per_hour: float) -> dict[str, Any]:
    """Split runs into Colab L4 and local 3070, sum train_wall_seconds, convert to hours and CU.

    A run needs `group`, `name` and `train_wall_seconds` (None when W&B has none); runs without it
    are listed under `no_train_wall_seconds` and contribute 0 h, never a guessed value.
    """
    out: dict[str, Any] = {}
    for label, local in (("colab_l4", False), ("local_3070", True)):
        members = [r for r in runs if (r["group"] in LOCAL_3070_GROUPS) == local]
        counted = [r for r in members if r["train_wall_seconds"] is not None]
        seconds = sum(r["train_wall_seconds"] for r in counted)
        out[label] = {
            "runs_counted": [r["name"] for r in counted],
            "runs_without_train_wall_seconds": [
                r["name"] for r in members if r["train_wall_seconds"] is None
            ],
            # W&B `_runtime` (process wall clock, a different quantity from train_wall_seconds),
            # listed for the runs that lack train_wall_seconds; NOT added to the totals.
            "wandb_runtime_seconds_of_those": {
                r["id"]: r.get("wandb_runtime_seconds")
                for r in members
                if r["train_wall_seconds"] is None
            },
            "train_wall_seconds": seconds,
            "hours": seconds / 3600.0,
        }
    out["colab_l4"]["cu_ESTIMATE"] = out["colab_l4"]["hours"] * cu_per_hour
    out["colab_l4"]["cu_per_hour_assumed"] = cu_per_hour
    return out


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_span(ref: str) -> dict[str, Any]:
    """First and last commit author dates reachable from `ref`, and the span in hours and days."""
    dates = _git(["log", ref, "--format=%aI"]).splitlines()
    if not dates:
        raise RuntimeError(f"git log {ref} returned no commits")
    first, last = min(dates, key=_iso), max(dates, key=_iso)
    hours = span_hours(first, last)
    return {
        "ref": ref,
        "ref_sha": _git(["rev-parse", ref]),
        "n_commits": len(dates),
        "first_commit_author_date": first,
        "last_commit_author_date": last,
        "span_hours": hours,
        "span_days": hours / 24.0,
        "command": f"git log {ref} --format=%aI  (min and max author date)",
    }


def read_json(rel: str, ref: str | None) -> dict[str, Any]:
    """Read a repo JSON from the working tree, else from `ref` via `git show`."""
    path = REPO_ROOT / rel
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if ref is None:
        raise FileNotFoundError(rel)
    return json.loads(_git(["show", f"{ref}:{rel}"]))


def fetch_wandb_runs() -> list[dict[str, Any]]:
    """Read-only W&B pull: id, name, group, state, created_at, heartbeat_at, train_wall_seconds."""
    import wandb  # lazy: needed only for the live pull

    runs = []
    for r in wandb.Api().runs(ENTITY_PROJECT):
        runs.append(
            {
                "id": r.id,
                "name": r.name,
                "group": r.group,
                "state": r.state,
                "created_at": r.created_at,
                "heartbeat_at": r.heartbeat_at,
                "train_wall_seconds": r.summary.get("train_wall_seconds"),
                "wandb_runtime_seconds": r.summary.get("_runtime"),
            }
        )
    return sorted(runs, key=lambda x: x["created_at"])


def crosscheck(runs: list[dict[str, Any]], files: dict[str, float]) -> dict[str, float]:
    """Absolute difference (s) between file-recorded and live W&B train_wall_seconds per run id."""
    live = {r["id"]: r["train_wall_seconds"] for r in runs}
    return {rid: abs(live[rid] - sec) for rid, sec in files.items() if rid in live}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("--ext-ref", default="origin/fix/overfit-watch-report")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/final/effort_compute.json")
    args = parser.parse_args(argv)

    runs = fetch_wandb_runs()
    summaries = read_json(SUMMARIES, None)["runs"]
    ext = read_json(EXT_SUMMARY, args.ext_ref)["runs"]
    rate = read_json(PILOT_SUMMARY, None)["colab_pro_compute_units_per_hour"]

    file_seconds = {v["id"]: v["train_wall_seconds"] for v in summaries.values()}
    file_seconds.update({v["wandb_run_id"]: v["train_wall_seconds_summary"] for v in ext.values()})
    diffs = crosscheck(runs, file_seconds)
    if len(diffs) != len(file_seconds) or max(diffs.values()) > 1e-6:
        raise RuntimeError(f"file and live W&B train_wall_seconds disagree: {diffs}")

    wandb_span_hours = span_hours(runs[0]["created_at"], max(r["heartbeat_at"] for r in runs))
    result = {
        "provenance": {
            "script": "scripts/compute_effort.py",
            "code_sha": _git(["rev-parse", "HEAD"]),
            "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wandb_project": ENTITY_PROJECT + " (private)",
            "wandb_access": "read-only Api().runs()",
            "files": [SUMMARIES, EXT_SUMMARY + f" (from {args.ext_ref} if absent)", PILOT_SUMMARY],
            "note": (
                "Human hands-on time ('about 8-9 hours') is the owner's figure, not computed here. "
                "Colab evaluation and final_all sessions are NOT included (not in W&B run "
                "summaries); that part stays {{FINAL_COLAB_HOURS}}. CU is hours x the owner's "
                "reported rate, an ESTIMATE."
            ),
        },
        "git_span": git_span(args.ref),
        "wandb_span": {
            "first_run_created_at": runs[0]["created_at"],
            "last_run_heartbeat_at": max(r["heartbeat_at"] for r in runs),
            "last_run_note": "heartbeat_at of the latest run, used as 'finished' (no finished_at)",
            "span_hours": wandb_span_hours,
            "n_runs": len(runs),
            "all_finished": all(r["state"] == "finished" for r in runs),
        },
        "gpu": compute_gpu_totals(runs, rate),
        "crosscheck_file_vs_live_wandb_abs_diff_seconds": diffs,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("git_span", "wandb_span", "gpu")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
