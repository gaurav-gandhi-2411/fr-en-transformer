from __future__ import annotations

# Benchmark: OLD naive bootstrap (per resample, rebuild the resampled hyp/ref lists and re-score
# them through official/score.py's own corpus_bleu / chrf_sentence, i.e. exactly what
# `score_slice` does per resample) vs NEW (per-sentence sufficient statistics computed once, then
# numpy gather+reduce; nmt.evaluate.bootstrap_ci_official). 1,000 resamples, seed 1234, per split.
# If the OLD run is projected to exceed OLD_BUDGET_S it is timed at OLD_PROBE_RESAMPLES resamples
# and linearly extrapolated (labelled "extrapolated" in the JSON). The OLD loop is linear in the
# number of resamples (each resample is independent identical work), which is why that is fair.
# Hypotheses: committed smoke predictions where they exist, else the references with a
# deterministic corruption (labelled in the JSON). Usage: `uv run python scripts/bench_bootstrap.py`
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from nmt.evaluate import (
    _official_metric_fn,
    bootstrap_ci_official,
    load_official_module,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT = REPO_ROOT / "reports" / "smoke_tuned" / "eval" / "export"
OUT = REPO_ROOT / "reports" / "bench_bootstrap.json"
N_RESAMPLES = 1000
SEED = 1234
OLD_BUDGET_S = 180.0  # ~3 min per split per metric
OLD_PROBE_RESAMPLES = 50
SPLITS = {"dev": "data/dev", "e1": "data/eval/e1", "e2": "data/eval/e2", "e3": "data/eval/e3"}
SPLITS["e2synth"] = "data/eval/e2synth"


def _cpu_model() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor()


def _load(split: str) -> tuple[list[str], list[str], str]:
    rows = [
        json.loads(x)
        for x in (REPO_ROOT / SPLITS[split] / "labels.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    refs = [r["reference"] for r in rows]
    pred_path = EXPORT / f"{split}_predictions.json"
    if pred_path.is_file():
        pred = json.loads(pred_path.read_text("utf-8"))
        return (
            [pred.get(r["id"], "") for r in rows],
            refs,
            f"smoke_tuned predictions ({pred_path.name})",
        )
    # Deterministic corruption: drop ~25% of words, seed 42 (no committed predictions here).
    rng = random.Random(42)
    hyps = [" ".join(w for w in r.split() if rng.random() > 0.25) for r in refs]
    return hyps, refs, "references with deterministic 25% word-drop corruption (seed 42)"


def _old_one_resample(metric: str, hyps: list[str], refs: list[str], row: list[int]) -> float:
    m = load_official_module()
    h = [hyps[i] for i in row]
    r = [refs[i] for i in row]
    if metric == "bleu":
        return m.corpus_bleu(h, r)
    return sum(m.chrf_sentence(a, b) for a, b in zip(h, r, strict=True)) / len(r)


def _time_old(metric: str, hyps: list[str], refs: list[str]) -> dict[str, object]:
    n = len(refs)
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_RESAMPLES, n)).tolist()
    t0 = time.perf_counter()
    _old_one_resample(metric, hyps, refs, idx[0])
    one = time.perf_counter() - t0
    if one * N_RESAMPLES > OLD_BUDGET_S:
        k = OLD_PROBE_RESAMPLES
        t0 = time.perf_counter()
        for row in idx[:k]:
            _old_one_resample(metric, hyps, refs, row)
        dt = time.perf_counter() - t0
        return {"seconds": dt * N_RESAMPLES / k, "extrapolated": True, "timed_resamples": k}
    t0 = time.perf_counter()
    for row in idx:
        _old_one_resample(metric, hyps, refs, row)
    return {
        "seconds": time.perf_counter() - t0,
        "extrapolated": False,
        "timed_resamples": N_RESAMPLES,
    }


def _exact_match_small(n: int = 60, resamples: int = 25) -> dict[str, object]:
    """NEW's resampled scores and CI vs OLD's, on the SAME indices, small n."""
    hyps, refs, _ = _load("e1")
    hyps, refs = hyps[:n], refs[:n]
    idx = np.random.default_rng(SEED).integers(0, n, size=(resamples, n))
    out: dict[str, object] = {"n": n, "resamples": resamples}
    for metric in ("bleu", "chrf"):
        fn = _official_metric_fn(metric)
        new = fn.resample_fn(fn.stats_fn(hyps, refs), idx)
        old = np.array([_old_one_resample(metric, hyps, refs, row) for row in idx.tolist()])
        out[f"{metric}_max_abs_diff"] = float(np.max(np.abs(new - old)))
        old_s = np.sort(old)
        ci = bootstrap_ci_official(hyps, refs, metric, resamples, SEED)  # same seed => same idx
        lo = min(float(old_s[int(0.025 * resamples)]), ci["point"])
        hi = max(float(old_s[min(resamples - 1, int(0.975 * resamples))]), ci["point"])
        out[f"{metric}_ci_max_abs_diff"] = max(abs(ci["ci_low"] - lo), abs(ci["ci_high"] - hi))
    return out


def main() -> None:
    result: dict[str, object] = {
        "machine": {
            "cpu": _cpu_model(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "old_definition": "per resample: rebuild resampled lists, official corpus_bleu / mean "
        "chrf_sentence (the components of official score_slice)",
        "new_definition": "nmt.evaluate.bootstrap_ci_official: stats once + numpy resample + CI",
        "exact_match_old_vs_new_same_indices": _exact_match_small(),
        "splits": {},
    }
    splits: dict[str, object] = {}
    for split in SPLITS:
        hyps, refs, source = _load(split)
        entry: dict[str, object] = {"n": len(refs), "hypotheses": source, "metrics": {}}
        for metric in ("bleu", "chrf"):
            t0 = time.perf_counter()
            ci = bootstrap_ci_official(hyps, refs, metric, N_RESAMPLES, SEED)
            new_s = time.perf_counter() - t0
            old = _time_old(metric, hyps, refs)
            entry["metrics"][metric] = {  # type: ignore[index]
                "old_seconds": round(float(old["seconds"]), 3),  # type: ignore[arg-type]
                "old_extrapolated": old["extrapolated"],
                "old_timed_resamples": old["timed_resamples"],
                "new_seconds": round(new_s, 3),
                "speedup": round(float(old["seconds"]) / new_s, 1),  # type: ignore[arg-type]
                "new_point": ci["point"],
                "new_ci": [ci["ci_low"], ci["ci_high"]],
            }
            print(split, metric, entry["metrics"][metric], flush=True)  # type: ignore[index]  # noqa: T201
        splits[split] = entry
    result["splits"] = splits
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
