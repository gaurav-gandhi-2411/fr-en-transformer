from __future__ import annotations

# Builds a tiny SYNTHETIC copy-task shard tree + a matching manifest.json under <repo>/data/shards
# so the notebook's smoke path (verify manifest -> train configs/smoke.yaml on CPU) can run in CI,
# where the real shards (gitignored, 145 MB, private HF dataset) are unavailable. This proves the
# notebook cells and the training subprocess execute on the target Python; it says NOTHING about
# model quality. Never run it in a tree that has real shards: it refuses to overwrite them.
#
# Usage: python colab/make_synthetic_shards.py <repo-root>
import hashlib
import json
import sys
from pathlib import Path

from nmt.data.synthetic import write_synthetic_split

VOCAB_SIZE = 16000  # configs/smoke.yaml model.vocab_size; synthetic ids must stay below it
# split name -> n_examples; the names mirror the real tree's trainable/eval splits.
SPLITS = {"train": 512, "e1": 64, "e2": 64, "e3": 64}


def build(repo_root: Path) -> Path:
    """Write synthetic shards + manifest.json into `<repo_root>/data/shards`; return that dir."""
    shards = repo_root / "data" / "shards"
    if (shards / "manifest.json").exists():
        raise RuntimeError(f"{shards} already has a manifest.json; refusing to overwrite.")
    for i, (name, n_examples) in enumerate(SPLITS.items()):
        write_synthetic_split(shards / name, n_examples, VOCAB_SIZE, seed=42 + i)
    spm = repo_root / "tokenizer" / "spm.model"
    manifest = {
        "vocab_size": VOCAB_SIZE,
        "synthetic": True,
        "spm_model_sha256": hashlib.sha256(spm.read_bytes()).hexdigest(),
        "shard_files_sha256": {
            f"{name}/shard_00000.npz": hashlib.sha256(
                (shards / name / "shard_00000.npz").read_bytes()
            ).hexdigest()
            for name in SPLITS
        },
    }
    (shards / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return shards


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: make_synthetic_shards.py <repo-root>", file=sys.stderr)
        return 2
    print(f"synthetic shards written to {build(Path(args[0]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
