from __future__ import annotations

# The recount rule on a tiny synthetic results dict, and the committed results.json against the
# README the generator renders from it (the two must state the same numbers).
import json
from pathlib import Path

from scripts import count_background_load as cbl

PROD = Path(__file__).resolve().parents[1] / "reports" / "final" / "production"


def _cell(pre: list[float | None], during: list[float | None]) -> dict:
    return {
        "background_cpu": {
            "per_run_pct": pre,
            "during_other_cpu_pct_per_run": during,
            "any_above_flag": any(v is not None and v > 30.0 for v in pre),
        }
    }


def test_strict_threshold_and_none_handling() -> None:
    res = {
        "config": {"background_flag_pct": 30.0},
        "latency_ms": {
            "a": _cell([30.0, 29.9], [None, None]),  # exactly 30.0 is NOT above
            "b": _cell([10.0, 30.1], [None, None]),
        },
        "throughput": {
            "a": _cell([5.0, 6.0], [None, 35.9]),  # only the during-run sample is above
            "b": _cell([5.0, 6.0], [None, None]),
        },
    }
    out = cbl.count_background_load(res)
    t = out["totals"]
    assert t["pre_latency"] == (1, 2) and t["pre_throughput"] == (0, 2)
    assert t["during_throughput"] == (1, 2) and t["during_latency"] == (0, 2)
    assert t["either_total"] == (2, 4)
    assert out["cells"]["latency:b"]["pre_runs"] == [2]
    assert out["cells"]["throughput:a"]["during_runs"] == [2]
    assert out["json_flag_agrees"]


def test_same_cell_keys_in_both_phases_are_not_merged() -> None:
    """Regression: latency and throughput share keys; the old README merged them in one dict."""
    res = {
        "config": {"background_flag_pct": 30.0},
        "latency_ms": {"k": _cell([40.0], [None])},
        "throughput": {"k": _cell([10.0], [None])},
    }
    assert cbl.count_background_load(res)["totals"]["pre_total"] == (1, 2)


def test_committed_results_and_readme_agree() -> None:
    res = json.loads((PROD / "results.json").read_text(encoding="utf-8"))
    out = cbl.count_background_load(res)
    assert out["json_flag_agrees"]
    lat, tp, tot = (out["totals"][f"pre_{k}"] for k in ("latency", "throughput", "total"))
    assert (lat[1], tp[1], tot[1]) == (12, 8, 20)
    readme = (PROD / "README.md").read_text(encoding="utf-8")
    assert (
        f"background load: {tot[0]} of 20 timing cells ({lat[0]} of 12 latency, {tp[0]} of 8 "
        "throughput)"
    ) in readme
