from __future__ import annotations

# Read-only audit of a finished W&B run (the Colab L4 main run). Pulls the FULL history with
# run.scan_history() (run.history() is sampled and would hide rows), plus config/summary/metadata,
# and derives every reported number from that data -- nothing is typed in. The pulled data is saved
# (history.jsonl.gz + run_meta.json) so `--from-file` re-derives the identical audit offline.
#
# W&B access is strictly read-only (wandb.Api() only; never wandb.init, never modifies a run).
#
# CLI:
#   online:  python scripts/audit_wandb_run.py --run ENTITY/PROJECT/ID --out audit.json
#            (also writes history.jsonl.gz and run_meta.json next to --out)
#   offline: python scripts/audit_wandb_run.py --from-file history.jsonl.gz --out audit.json
#            (reads run_meta.json next to the history file unless --meta is given)
import argparse
import gzip
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# GG's Colab Pro rate for the L4 (compute units per hour). An ESTIMATE, not a billed figure.
CU_PER_HOUR_L4 = 1.54
PCTS = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
PROVENANCE_KEYS = ("git_sha", "git_dirty", "precision", "precision_requested", "gpu_name")
EVAL_PREFIX = "eval/"
# Table-valued eval key (W&B Table payload, not a scalar series).
NON_SERIES = {"eval/sample_translations"}


def _num(x: Any) -> float | None:
    """`x` as a float if it is a real number (bool excluded), else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _finite(x: Any) -> bool:
    v = _num(x)
    return v is not None and math.isfinite(v)


def train_rows(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows carrying a training `loss` (key may be present-but-None on eval-only rows)."""
    return [r for r in history if "loss" in r and r.get("loss") is not None]


def loss_curve(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """First/last/min loss, loss nearest 10/25/50/75/90/100% of the final step, smoothed tail."""
    rows = [
        r for r in train_rows(history) if _finite(r["loss"]) and _num(r.get("step")) is not None
    ]
    if not rows:
        return {"n_rows": 0}
    steps = sorted(int(r["step"]) for r in rows)
    last_step = max(steps)
    at: dict[str, Any] = {}
    for p in PCTS:
        tgt = p * last_step
        row = min(rows, key=lambda r: (abs(r["step"] - tgt), r["step"]))
        at[f"{int(p * 100)}%"] = {"target_step": tgt, "step": int(row["step"]), "loss": row["loss"]}
    tail_rows = [r for r in rows if r["step"] > last_step - 1000]
    best = min(rows, key=lambda r: r["loss"])
    return {
        "n_rows": len(rows),
        "first": {"step": int(rows[0]["step"]), "loss": rows[0]["loss"]},
        "last": {"step": int(rows[-1]["step"]), "loss": rows[-1]["loss"]},
        "min": {"step": int(best["step"]), "loss": best["loss"]},
        "loss_at_pct_of_final_step": at,
        "last_1000_step_mean": {
            "mean": statistics.fmean(r["loss"] for r in tail_rows),
            "n_rows": len(tail_rows),
            "step_range": [int(tail_rows[0]["step"]), int(tail_rows[-1]["step"])],
        },
        "observed_log_every_steps": sorted({b - a for a, b in zip(steps, steps[1:], strict=False)}),
    }


def eval_points(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that carry an eval/val_loss, each with all numeric eval/* values."""
    out = []
    for r in history:
        if r.get(EVAL_PREFIX + "val_loss") is None:
            continue
        pt: dict[str, Any] = {"step": r.get("step", r.get("_step"))}
        for k, v in r.items():
            if k.startswith(EVAL_PREFIX) and k not in NON_SERIES and _num(v) is not None:
                pt[k] = v
        out.append(pt)
    return out


def rise_runs(
    values: Sequence[float], steps: Sequence[int], min_rises: int = 2
) -> list[dict[str, Any]]:
    """Maximal runs of strictly increasing consecutive values with >= `min_rises` rises."""
    runs: list[dict[str, Any]] = []
    i = 0
    while i < len(values) - 1:
        if values[i + 1] > values[i]:
            j = i
            while j < len(values) - 1 and values[j + 1] > values[j]:
                j += 1
            if j - i >= min_rises:
                runs.append(
                    {
                        "n_rises": j - i,
                        "steps": list(steps[i : j + 1]),
                        "values": list(values[i : j + 1]),
                        "start_index": i,
                    }
                )
            i = j
        else:
            i += 1
    return runs


def val_loss_report(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """E1 validation loss series, min/last, and the overfitting (consecutive-rise) check."""
    pts = [p for p in points if _finite(p.get("eval/val_loss"))]
    if not pts:
        return {"n_evals": 0}
    steps = [int(p["step"]) for p in pts]
    vals = [p["eval/val_loss"] for p in pts]
    imin = min(range(len(vals)), key=lambda i: (vals[i], i))
    runs = rise_runs(vals, steps)
    after = [r for r in runs if r["start_index"] >= imin]
    n_rises_after = sum(1 for a, b in zip(vals[imin:], vals[imin + 1 :], strict=False) if b > a)
    return {
        "n_evals": len(vals),
        "series": [{"step": s, "val_loss": v} for s, v in zip(steps, vals, strict=True)],
        "min": {"step": steps[imin], "val_loss": vals[imin]},
        "last": {"step": steps[-1], "val_loss": vals[-1]},
        "last_minus_min": vals[-1] - vals[imin],
        "evals_after_min": len(vals) - 1 - imin,
        "rises_after_min": n_rises_after,
        "rise_runs_ge2_consecutive": runs,
        "rise_runs_ge2_consecutive_after_min": after,
        "overfit_ge2_consecutive_rises_after_min": bool(after),
    }


def eval_series(points: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Every eval/* scalar metric as a [{step, value}] series."""
    keys = sorted({k for p in points for k in p if k.startswith(EVAL_PREFIX)})
    return {k: [{"step": p["step"], "value": p[k]} for p in points if k in p] for k in keys}


def stability(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Non-finite / skipped steps, grad_skip_count trajectory, loss_scale values."""
    rows = train_rows(history)
    bad_loss = [r["step"] for r in rows if not _finite(r.get("loss"))]
    bad_gn = [
        r["step"] for r in rows if r.get("grad_norm") is not None and not _finite(r["grad_norm"])
    ]
    missing_gn = [r["step"] for r in rows if r.get("grad_norm") is None]
    skips = [
        (r["step"], r["grad_skip_count"])
        for r in rows
        if _num(r.get("grad_skip_count")) is not None
    ]
    incr = [
        {"step": b[0], "from": a[1], "to": b[1]}
        for a, b in zip(skips, skips[1:], strict=False)
        if b[1] > a[1]
    ]
    scales = [r.get("loss_scale") for r in rows if r.get("loss_scale") is not None]
    scale_changes = [
        {"step": b["step"], "from": a.get("loss_scale"), "to": b.get("loss_scale")}
        for a, b in zip(rows, rows[1:], strict=False)
        if a.get("loss_scale") != b.get("loss_scale")
    ]
    gns = [r["grad_norm"] for r in rows if _finite(r.get("grad_norm"))]
    return {
        "n_train_rows": len(rows),
        "nonfinite_loss_rows": len(bad_loss),
        "nonfinite_loss_steps": bad_loss,
        "nonfinite_grad_norm_rows": len(bad_gn),
        "nonfinite_grad_norm_steps": bad_gn,
        "missing_grad_norm_rows": len(missing_gn),
        "grad_norm": {"max": max(gns), "median": statistics.median(gns)} if gns else None,
        "grad_skip_count": {
            "max": max((s[1] for s in skips), default=None),
            "last": skips[-1][1] if skips else None,
            "increments_between_logged_rows": incr,
        },
        "optimizer_stepped_false_rows": sum(1 for r in rows if r.get("optimizer_stepped") is False),
        "optimizer_stepped_missing_rows": sum(1 for r in rows if "optimizer_stepped" not in r),
        "loss_scale_distinct": sorted(set(scales)),
        "loss_scale_changes": scale_changes,
    }


def step_anomalies(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Visible resumes: step resets (decreasing step in row order) and duplicate train steps."""
    steps = [int(r["step"]) for r in train_rows(history)]
    resets = [
        {"index": i + 1, "from_step": a, "to_step": b}
        for i, (a, b) in enumerate(zip(steps, steps[1:], strict=False))
        if b < a
    ]
    dups = sorted({s for s in set(steps) if steps.count(s) > 1})
    return {"step_resets": resets, "duplicate_train_steps": dups}


def wall_report(
    history: Sequence[dict[str, Any]], summary: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Runtime, wall_step_s sum over logged rows, tok/s median, and the CU ESTIMATE."""
    rows = train_rows(history)
    wss = [r["wall_step_s"] for r in rows if _finite(r.get("wall_step_s"))]
    tps = [r["tok_per_sec"] for r in rows if _finite(r.get("tok_per_sec"))]
    runtime_s = _num(summary.get("_runtime"))
    if runtime_s is None:
        runtime_s = _num((summary.get("_wandb") or {}).get("runtime"))
    hours = runtime_s / 3600 if runtime_s is not None else None
    return {
        "runtime_seconds_summary": runtime_s,
        "runtime_hours": hours,
        "train_wall_seconds_summary": summary.get("train_wall_seconds"),
        "wall_step_s_sum_logged_rows": sum(wss) if wss else None,
        "wall_step_s_rows": len(wss),
        "wall_step_s_median": statistics.median(wss) if wss else None,
        "wall_step_s_note": "logged rows only (every log_every-th step), so the sum undercounts",
        "tok_per_sec_median": statistics.median(tps) if tps else None,
        "metadata_startedAt": metadata.get("startedAt"),
        "cu_used_ESTIMATE": {
            "cu": hours * CU_PER_HOUR_L4 if hours is not None else None,
            "formula": f"runtime_hours x {CU_PER_HOUR_L4} CU/h (GG's Colab Pro L4 rate)",
            "is_estimate": True,
        },
    }


def provenance(
    history: Sequence[dict[str, Any]],
    info: dict[str, Any],
    config: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Run state/times, config provenance keys, step/epoch accounting."""

    def cfgv(k: str) -> Any:
        v = config.get(k)
        return v.get("value") if isinstance(v, dict) and "value" in v else v

    def sub(k: str, k2: str) -> Any:
        d = cfgv(k)
        return d.get(k2) if isinstance(d, dict) else None

    rows = train_rows(history)
    last = max(rows, key=lambda r: r["step"]) if rows else {}
    return {
        "run": info,
        "config": {k: cfgv(k) for k in PROVENANCE_KEYS}
        | {k: cfgv(k) for k in sorted(config) if k.startswith("preflight_")},
        "planned_steps": sub("optim", "planned_steps"),
        "log_every_config": sub("logging", "log_every"),
        "eval_every_config": sub("eval", "eval_every"),
        "eval_e1_n_e2_n_e3_n_config": [sub("eval", k) for k in ("e1_n", "e2_n", "e3_n")],
        "resume_count": {"config": cfgv("resume_count"), "summary": summary.get("resume_count")},
        "wait_seconds_total": {
            "config": cfgv("wait_seconds_total"),
            "summary": summary.get("wait_seconds_total"),
        },
        "train_wall_seconds": summary.get("train_wall_seconds"),
        "final_step_summary": summary.get("final_step"),
        "last_logged_step": last.get("step"),
        "epoch_fraction_at_last_logged_step": last.get("epoch_fraction"),
        "epoch_fraction_summary": summary.get("epoch_fraction"),
        "exit_reason": summary.get("exit_reason"),
        "n_history_rows": len(history),
        "n_train_rows": len(rows),
        "n_eval_rows": sum(1 for r in history if r.get(EVAL_PREFIX + "val_loss") is not None),
    }


def audit(history: Sequence[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Full audit dict from the pulled history and run meta (info/config/summary/metadata)."""
    summary = meta.get("summary") or {}
    pts = eval_points(history)
    return {
        "provenance": provenance(
            history, meta.get("info") or {}, meta.get("config") or {}, summary
        ),
        "train_loss": loss_curve(history),
        "val_loss_e1": val_loss_report(pts),
        "eval_series": eval_series(pts),
        "stability": stability(history),
        "step_anomalies": step_anomalies(history),
        "wall": wall_report(history, summary, meta.get("metadata") or {}),
    }


def _jsonable(x: Any) -> Any:
    """Best-effort conversion of W&B API objects to plain JSON types."""
    return json.loads(json.dumps(x, default=str))


def pull(run_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read-only pull of the full history and run meta via wandb.Api()."""
    import wandb  # lazy: only the online path needs it

    run = wandb.Api().run(run_path)
    history = [_jsonable(dict(r)) for r in run.scan_history()]
    info = {
        "path": run_path,
        "state": run.state,
        "created_at": run.created_at,
        "heartbeat_at": getattr(run, "heartbeat_at", None),
        "name": run.name,
        "url": run.url,
    }
    meta = {
        "info": info,
        "config": _jsonable(dict(run.config)),
        "summary": _jsonable(dict(run.summary)),
        "metadata": _jsonable(getattr(run, "metadata", None) or {}),
    }
    return history, meta


def write_history(path: Path, history: Iterable[dict[str, Any]]) -> None:
    """Write history rows as gzip-compressed JSON lines."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in history:
            f.write(json.dumps(r) + "\n")


def read_history(path: Path) -> list[dict[str, Any]]:
    """Read history rows from a .jsonl or .jsonl.gz file."""
    opener: Any = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_markdown(a: dict[str, Any]) -> str:
    """Short human table of the headline numbers."""
    p, t, v, s, w = a["provenance"], a["train_loss"], a["val_loss_e1"], a["stability"], a["wall"]
    cu = w["cu_used_ESTIMATE"]
    rows = [
        ("run state", p["run"].get("state")),
        ("git_sha / dirty", f"{p['config'].get('git_sha')} / {p['config'].get('git_dirty')}"),
        ("precision / gpu", f"{p['config'].get('precision')} / {p['config'].get('gpu_name')}"),
        ("planned_steps / last logged step", f"{p['planned_steps']} / {p['last_logged_step']}"),
        ("final_step (summary)", p["final_step_summary"]),
        ("epoch_fraction at last logged step", p["epoch_fraction_at_last_logged_step"]),
        (
            "history rows (train / eval)",
            f"{p['n_history_rows']} ({p['n_train_rows']} / {p['n_eval_rows']})",
        ),
        (
            "resume_count / wait_seconds_total",
            f"{p['resume_count']['summary']} / {p['wait_seconds_total']['summary']}",
        ),
        ("train loss first -> last", f"{t['first']['loss']:.4f} -> {t['last']['loss']:.4f}"),
        ("train loss min", f"{t['min']['loss']:.4f} @ {t['min']['step']}"),
        ("train loss last-1000 mean", f"{t['last_1000_step_mean']['mean']:.4f}"),
        ("val_loss min", f"{v['min']['val_loss']:.4f} @ {v['min']['step']}"),
        ("val_loss last", f"{v['last']['val_loss']:.4f} @ {v['last']['step']}"),
        ("val_loss >=2 consecutive rises after min", v["overfit_ge2_consecutive_rises_after_min"]),
        (
            "non-finite loss / grad_norm rows",
            f"{s['nonfinite_loss_rows']} / {s['nonfinite_grad_norm_rows']}",
        ),
        ("grad_skip_count max", s["grad_skip_count"]["max"]),
        ("optimizer_stepped False rows", s["optimizer_stepped_false_rows"]),
        ("loss_scale distinct", s["loss_scale_distinct"]),
        ("runtime hours", w["runtime_hours"]),
        ("tok/s median", w["tok_per_sec_median"]),
        ("CU used (ESTIMATE)", f"{cu['cu']} ({cu['formula']})"),
    ]
    lines = [
        "# Main L4 run audit (derived by scripts/audit_wandb_run.py)",
        "",
        "| item | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {val} |" for k, val in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only audit of a finished W&B run.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="entity/project/id (pulls via wandb.Api, read-only)")
    src.add_argument("--from-file", type=Path, help="history.jsonl[.gz] saved by a previous pull")
    ap.add_argument("--meta", type=Path, default=None, help="run_meta.json (default: sibling)")
    ap.add_argument("--out", type=Path, required=True, help="output audit JSON")
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.run:
        history, meta = pull(args.run)
        write_history(args.out.parent / "history.jsonl.gz", history)
        (args.out.parent / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        history = read_history(args.from_file)
        meta_path = args.meta or args.from_file.parent / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    result = audit(history, meta)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(f"wrote {args.out} ({len(history)} history rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
