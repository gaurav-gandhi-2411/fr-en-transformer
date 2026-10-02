from __future__ import annotations

# COMET-22 for the final evaluation of the 4 runs (reports/final). Scores every distinct
# (src, mt, ref) triple across all runs x variants (seg_off, seg_tuned) x splits ONCE, in
# resumable chunks (one `envs/comet` process per chunk), then writes
# `<out-root>/<run>/<variant>/comet.json` with the per-sentence scores in SPLITS order and the
# per-split system means. Identical triples get identical COMET scores (deterministic, no sampling),
# so scoring each distinct triple once covers exactly the same sentences as scoring all of them;
# the dedup counts are recorded in every comet.json. Eval-only: never feeds selection.
#
# CLI: `python -m scripts.comet_final --out-root reports/final --work-dir <scratch> --gpus 0
#   [--chunk-size 2500] [--max-chunks N]`   (rerun the same command to resume)
import argparse
import json
from pathlib import Path
from typing import Any

from nmt.evaluate import load_split, run_comet

RUNS = ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
VARIANTS = ("seg_off", "seg_tuned")
SPLITS = ("dev", "e1", "e2", "e2synth", "e3")
Triple = tuple[str, str, str]


def collect_triples(out_root: Path) -> tuple[list[Triple], dict[tuple[str, str, str], list[int]]]:
    """(unique triples in first-occurrence order, {(run, variant, split): index per sentence}).

    Sentence order inside a split is the split's inputs order, the order every report uses."""
    unique: list[Triple] = []
    position: dict[Triple, int] = {}
    layout: dict[tuple[str, str, str], list[int]] = {}
    for split in SPLITS:
        inputs, labels = load_split(split)
        ref_by_id = {r["id"]: r["reference"] for r in labels}
        for run in RUNS:
            for variant in VARIANTS:
                pred = json.loads(
                    (out_root / run / variant / f"{split}_predictions.json").read_text(
                        encoding="utf-8"
                    )
                )
                idx = []
                for r in inputs:
                    t = (r["source"], pred[r["id"]], ref_by_id[r["id"]])
                    if t not in position:
                        position[t] = len(unique)
                        unique.append(t)
                    idx.append(position[t])
                layout[(run, variant, split)] = idx
    return unique, layout


def split_chunks(n: int, chunk_size: int) -> list[tuple[int, int]]:
    """[start, end) bounds covering range(n) in order, each at most chunk_size long."""
    return [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]


def assemble(
    scores: list[float], layout: dict[tuple[str, str, str], list[int]], run: str, variant: str
) -> dict[str, Any]:
    """Per-split per-sentence scores and system means for one (run, variant)."""
    per_split = {s: [scores[i] for i in layout[(run, variant, s)]] for s in SPLITS}
    return {
        "per_split_scores": per_split,
        "system_score_by_split": {s: sum(v) / len(v) for s, v in per_split.items()},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="COMET-22 over the final evaluation outputs.")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True, help="scratch dir for chunk results")
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=2500)
    p.add_argument("--max-chunks", type=int, default=None, help="stop after N new chunks")
    args = p.parse_args(argv)
    unique, layout = collect_triples(args.out_root)
    chunks = split_chunks(len(unique), args.chunk_size)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    done_now = 0
    for k, (lo, hi) in enumerate(chunks):
        out = args.work_dir / f"chunk_{k:03d}.json"
        if out.is_file():
            continue
        if args.max_chunks is not None and done_now >= args.max_chunks:
            print("comet_final: stopping at --max-chunks; rerun to resume")
            return 0
        triples = [{"src": s, "mt": m, "ref": r} for s, m, r in unique[lo:hi]]
        run_comet(triples, out, timeout_seconds=6 * 3600.0, gpus=args.gpus)
        done_now += 1
        print(f"comet_final: chunk {k + 1}/{len(chunks)} done ({hi}/{len(unique)})", flush=True)
    results = [
        json.loads((args.work_dir / f"chunk_{k:03d}.json").read_text(encoding="utf-8"))
        for k in range(len(chunks))
    ]
    scores = [x for r in results for x in r["scores"]]
    if len(scores) != len(unique):
        raise RuntimeError(f"{len(scores)} scores for {len(unique)} distinct triples")
    total = sum(len(v) for v in layout.values())
    meta = {
        "model": results[0]["model"],
        "model_class": results[0]["model_class"],
        "class_identifier": results[0]["class_identifier"],
        "gpus": args.gpus,
        "device": "cpu" if args.gpus == 0 else "gpu",
        "n_distinct_triples_scored": len(unique),
        "n_triples_total_all_runs_variants": total,
        "dedup": "identical (src, mt, ref) triples are scored once and share the score",
        "chunk_wall_seconds": [r["wall_seconds"] for r in results],
    }
    for run in RUNS:
        for variant in VARIANTS:
            out = {"run": run, "variant": variant, **meta, **assemble(scores, layout, run, variant)}
            all_scores = [x for s in SPLITS for x in out["per_split_scores"][s]]
            out["n"] = len(all_scores)
            out["system_score"] = sum(all_scores) / len(all_scores)
            path = args.out_root / run / variant / "comet.json"
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"comet_final: wrote comet.json for {len(RUNS) * len(VARIANTS)} (run, variant) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
