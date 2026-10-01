from __future__ import annotations

# Paired A/B comparison CLI (spec §8: "paired bootstrap resampling (Koehn 2004) for every A/B
# comparison (ablations, decoding options), reporting Delta, CI and p"). Reads the
# `<split>_predictions.json` files two `evaluate` runs already wrote, so a comparison never
# re-decodes and always scores exactly the predictions behind each run's eval.json. Delta is
# A - B per split (official BLEU and chrF), plus per official slice for dev.
#
# CLI: `python -m nmt.compare --a DIR_A --b DIR_B --out compare.json
#   [--splits dev e1 e2 e3] [--n-bootstrap 1000] [--seed 1234]`
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from nmt.evaluate import _official_metric_fn, load_split, paired_bootstrap

DEFAULT_SPLITS: tuple[str, ...] = ("dev", "e1", "e2", "e3")
METRICS: tuple[str, ...] = ("bleu", "chrf")


def _load_preds(eval_dir: Path, split: str) -> dict[str, str]:
    path = Path(eval_dir) / f"{split}_predictions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _paired(
    hyps_a: list[str], hyps_b: list[str], refs: list[str], n_bootstrap: int, seed: int
) -> dict[str, dict[str, Any]]:
    return {
        m: paired_bootstrap(hyps_a, hyps_b, refs, _official_metric_fn(m), n_bootstrap, seed)
        for m in METRICS
    }


def compare_eval_dirs(
    dir_a: Path,
    dir_b: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    n_bootstrap: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Paired bootstrap of run A vs run B on every split in `splits`. Both runs must cover the
    split's full id set -- a missing id is an error, not a silently empty hypothesis."""
    result: dict[str, Any] = {
        "a": str(dir_a),
        "b": str(dir_b),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "splits": {},
    }
    for split in splits:
        _, labels = load_split(split)
        pred_a, pred_b = _load_preds(dir_a, split), _load_preds(dir_b, split)
        ids = [r["id"] for r in labels]
        missing = [i for i in ids if i not in pred_a or i not in pred_b]
        if missing:
            raise ValueError(f"{split}: {len(missing)} ids missing from A or B, e.g. {missing[:3]}")
        refs = [r["reference"] for r in labels]
        entry: dict[str, Any] = {
            "n": len(ids),
            "overall": _paired(
                [pred_a[i] for i in ids], [pred_b[i] for i in ids], refs, n_bootstrap, seed
            ),
        }
        by_slice: dict[str, list[int]] = defaultdict(list)
        for k, row in enumerate(labels):
            by_slice[row["slice"]].append(k)
        if len(by_slice) > 1:
            entry["by_slice"] = {
                name: _paired(
                    [pred_a[ids[k]] for k in idx],
                    [pred_b[ids[k]] for k in idx],
                    [refs[k] for k in idx],
                    n_bootstrap,
                    seed,
                )
                for name, idx in by_slice.items()
            }
        result["splits"][split] = entry
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap A/B comparison (spec §8).")
    parser.add_argument("--a", required=True, type=Path, help="Eval dir of system A.")
    parser.add_argument("--b", required=True, type=Path, help="Eval dir of system B.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    result = compare_eval_dirs(args.a, args.b, tuple(args.splits), args.n_bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, entry in result["splits"].items():
        c = entry["overall"]["chrf"]
        print(
            f"{split}: chrF delta={c['delta']:+.3f} "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
