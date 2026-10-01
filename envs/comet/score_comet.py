from __future__ import annotations

# COMET-22 (Unbabel/wmt22-comet-da) scorer, run in this isolated uv project because
# unbabel-comet pins numpy<2 / old torchmetrics and does not co-resolve with the main repo's
# pinned torch==2.14.0/numpy==2.5.3. Invoked by nmt.evaluate.run_comet via
# `uv run --project envs/comet python envs/comet/score_comet.py --in X --out Y` (spec §8).
# Eval-only: this script and its output are never consumed by nmt/selection.py.
import argparse
import json
import time
from pathlib import Path

from comet import download_model, load_from_checkpoint

MODEL_NAME = "Unbabel/wmt22-comet-da"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score {src,mt,ref} triples with COMET-22.")
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    triples = json.loads(args.in_path.read_text(encoding="utf-8"))
    data = [{"src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triples]

    t0 = time.monotonic()
    ckpt_path = download_model(MODEL_NAME)
    model = load_from_checkpoint(ckpt_path)
    output = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
    wall_seconds = time.monotonic() - t0

    result = {
        "model": MODEL_NAME,
        "n": len(data),
        "scores": list(output.scores),
        "system_score": float(output.system_score),
        "wall_seconds": wall_seconds,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"COMET: n={result['n']} system_score={result['system_score']:.4f} wall={wall_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
