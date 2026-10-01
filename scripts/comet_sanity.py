from __future__ import annotations

# COMET-22 sanity check (audit item: the earlier "0.929 on 3 sentences" figure has no saved
# inputs or output, so it cannot be traced and is withdrawn). Scores 50 official-dev sentences
# (stratified 20 seen / 10 long / 20 unseen_domain, seeded) three ways in one COMET call:
#   model     -- the given system's predictions vs the true references (the real measurement);
#   shuffled  -- the same predictions vs references deranged across sentences;
#   oracle    -- the reference itself as the hypothesis (upper bound).
# A reference-based model must score `shuffled` and `oracle` differently from `model`; a
# referenceless (QE) model would ignore the reference entirely. Writes one JSON with per-condition
# system scores and the model class COMET reports.
#
# CLI: `python scripts/comet_sanity.py --pred reports/smoke_tuned/eval/export/dev_predictions.json
#   --out reports/audit/comet_sanity.json [--seed 1234]`
import argparse
import json
import random
import statistics
from pathlib import Path

from nmt.evaluate import load_split, run_comet

STRATA: dict[str, int] = {"seen": 20, "long": 10, "unseen_domain": 20}


def sample_ids(labels: list[dict], seed: int) -> list[str]:
    rng = random.Random(seed)
    chosen: list[str] = []
    for slice_name, k in STRATA.items():
        ids = sorted(r["id"] for r in labels if r["slice"] == slice_name)
        chosen.extend(sorted(rng.sample(ids, k)))
    return chosen


def derange(items: list[str], seed: int) -> list[str]:
    """Rotate by one after a seeded shuffle of positions: no item stays in its own slot."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    out = list(items)
    for a, b in zip(order, order[1:] + order[:1], strict=True):
        out[a] = items[b]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    inputs, labels = load_split("dev")
    src_by_id = {r["id"]: r["source"] for r in inputs}
    ref_by_id = {r["id"]: r["reference"] for r in labels}
    preds = json.loads(args.pred.read_text(encoding="utf-8"))
    ids = sample_ids(labels, args.seed)
    srcs = [src_by_id[i] for i in ids]
    refs = [ref_by_id[i] for i in ids]
    mts = [preds[i] for i in ids]
    shuffled_refs = derange(refs, args.seed)

    conditions = {
        "model": [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs, strict=True)],
        "shuffled": [
            {"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, shuffled_refs, strict=True)
        ],
        "oracle": [{"src": s, "mt": r, "ref": r} for s, r in zip(srcs, refs, strict=True)],
    }
    triples = [t for name in conditions for t in conditions[name]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = run_comet(triples, args.out.with_suffix(".raw.json"))

    n = len(ids)
    result: dict = {
        "pred_file": str(args.pred),
        "seed": args.seed,
        "n_per_condition": n,
        "strata": STRATA,
        "ids": ids,
        "comet_model": raw["model"],
        "comet_model_class": raw.get("model_class"),
        "comet_class_identifier": raw.get("class_identifier"),
        "comet_wall_seconds": raw["wall_seconds"],
        "conditions": {},
    }
    for k, name in enumerate(conditions):
        scores = raw["scores"][k * n : (k + 1) * n]
        result["conditions"][name] = {"system_score": statistics.fmean(scores), "scores": scores}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, c in result["conditions"].items():
        print(f"{name}: {c['system_score']:.4f}")
    print(f"model_class={result['comet_model_class']} id={result['comet_class_identifier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
