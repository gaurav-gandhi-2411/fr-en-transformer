from __future__ import annotations

# Quick post-training greedy eval on dev (the notebook's RUN_EVAL cell), run as a SUBPROCESS so
# the project code and the packages pip upgraded are imported in a fresh interpreter, not in the
# Colab kernel. Exports the latest checkpoint(s) and decodes with beam=1 + the official scorer.
#
# Usage (cwd = repo root): python colab/quick_eval.py <config-name> <run-dir>
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: quick_eval.py <config-name> <run-dir>", file=sys.stderr)
        return 2
    config_name, run_root = args[0], Path(args[1])
    sys.path.insert(0, str(Path.cwd()))
    from nmt.pipeline import stage_evaluate, stage_export

    export_dir = run_root / "export"
    stage_export(
        Path.cwd() / "configs" / f"{config_name}.yaml", out_dir=export_dir, run_dir=run_root
    )
    eval_dir = run_root / "eval"
    eval_result = stage_evaluate(
        export_dir,
        run_name=config_name,
        ckpt_name="export",
        seed=1234,
        beam=1,  # greedy, per the "quick eval" scope
        splits=("dev",),
        out_dir=eval_dir,
    )
    print(f"quick eval (greedy, dev): wrote {eval_dir / 'eval.json'}")
    for split_name, entry in eval_result["sets"].items():
        official = entry["official"]
        print(f"  {split_name}: OVERALL={official.get('OVERALL')}")
        for slice_name, slice_scores in official.get("by_slice", {}).items():
            print(f"    slice={slice_name}: {slice_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
