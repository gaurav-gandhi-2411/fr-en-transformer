from __future__ import annotations

# Deterministic recount of the production-benchmark background-load statement, straight from
# reports/final/production/results.json (no model, no network).
#
# RULES (stated exactly; threshold read from config.background_flag_pct = 30.0):
#   pre-run : a RUN counts if its pre-run background CPU % (`background_cpu.per_run_pct[i]`, a 5 s
#             machine-wide sample taken right before the run started) is STRICTLY > 30. A CELL
#             counts if any of its runs counts (this equals `background_cpu.any_above_flag`).
#   during  : separately, a run counts if its during-run other-process CPU %
#             (`background_cpu.during_other_cpu_pct_per_run[i]`, recorded only for the repeat
#             repetition, null elsewhere) is STRICTLY > 30. A cell counts if any run counts.
#   either  : pre-run OR during (the union), for reference.
# Cells: 12 latency cells (2 thread settings x 2 precisions x greedy/beam5/beam5_noseg) and
# 8 throughput cells (x greedy/beam5). Cells of one (phase, threads, precision) group share the
# same jobs, so a single noisy job flags several cells.
#
# Usage: python -m scripts.count_background_load [path/to/results.json]
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_RESULTS = Path("reports/final/production/results.json")


def _flag_runs(values: list[float | None], threshold: float) -> list[int]:
    """1-based indices of values strictly above `threshold` (None = not recorded, never counts)."""
    return [i + 1 for i, v in enumerate(values) if v is not None and v > threshold]


def count_background_load(res: dict[str, Any]) -> dict[str, Any]:
    """Per-cell and total counts under the pre-run / during / either rules."""
    threshold = float(res["config"]["background_flag_pct"])
    out: dict[str, Any] = {"threshold_pct": threshold, "cells": {}}
    for phase, section in (("latency", "latency_ms"), ("throughput", "throughput")):
        for key, cell in res[section].items():
            bg = cell["background_cpu"]
            out["cells"][f"{phase}:{key}"] = {
                "phase": phase,
                "pre_runs": _flag_runs(bg["per_run_pct"], threshold),
                "during_runs": _flag_runs(bg["during_other_cpu_pct_per_run"], threshold),
                "any_above_flag_in_json": bg["any_above_flag"],
            }
    totals: dict[str, tuple[int, int]] = {}
    for rule in ("pre", "during", "either"):
        for phase in ("latency", "throughput"):
            cells = [c for c in out["cells"].values() if c["phase"] == phase]
            if rule == "pre":
                hit = [c for c in cells if c["pre_runs"]]
            elif rule == "during":
                hit = [c for c in cells if c["during_runs"]]
            else:
                hit = [c for c in cells if c["pre_runs"] or c["during_runs"]]
            totals[f"{rule}_{phase}"] = (len(hit), len(cells))
        lat, tp = totals[f"{rule}_latency"], totals[f"{rule}_throughput"]
        totals[f"{rule}_total"] = (lat[0] + tp[0], lat[1] + tp[1])
    out["totals"] = totals
    # The JSON's own flag must agree with the pre-run rule it documents.
    out["json_flag_agrees"] = all(
        bool(c["pre_runs"]) == c["any_above_flag_in_json"] for c in out["cells"].values()
    )
    return out


def render(counts: dict[str, Any]) -> str:
    """Human-readable report of `count_background_load` output."""
    t = counts["totals"]
    lines = [f"threshold: strictly > {counts['threshold_pct']:.0f}% total CPU"]
    labels = {
        "pre": "pre-run sample > 30% (the README / model-card rule)",
        "during": "during-run other-process sample > 30% (recorded for the 4th repetition only)",
        "either": "pre-run OR during-run > 30%",
    }
    for rule, label in labels.items():
        lat, tp, tot = t[f"{rule}_latency"], t[f"{rule}_throughput"], t[f"{rule}_total"]
        lines.append(f"{label}:")
        lines.append(
            f"  latency {lat[0]} of {lat[1]}; throughput {tp[0]} of {tp[1]}; "
            f"total {tot[0]} of {tot[1]}"
        )
    lines.append("per cell (1-based run indices above the threshold):")
    for key, c in counts["cells"].items():
        lines.append(f"  {key}: pre-run {c['pre_runs']}, during {c['during_runs']}")
    lines.append(f"results.json any_above_flag == pre-run rule: {counts['json_flag_agrees']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_RESULTS
    res = json.loads(path.read_text(encoding="utf-8"))
    print(render(count_background_load(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
