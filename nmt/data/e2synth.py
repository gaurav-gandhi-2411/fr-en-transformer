from __future__ import annotations

# E2-synth: a SYNTHETIC long-input probe built by concatenating 2-4 randomly chosen E2 pairs
# (fr joined with one space, en joined with one space) so that source length reaches a regime
# natural data barely covers (E2 median 270 chars, p90 420, only a handful above 800). It exists
# to check how translation quality and the segmentation fallback behave as the French source
# grows from 400 to 900 characters.
#
# It is EVALUATION ONLY and reporting-only. It reuses E2's sentences -- E2 being the
# decoding-tuning set -- so it measures *length generalization*, not held-out content,
# and must never be reachable from checkpoint/decoding selection: nmt/selection.py and
# nmt/tune.py do not know it exists (tests/test_selection.py scans both for any `synth` reference).
#
# Output (same file format as the other E-sets, plus `"synthetic": true` on every row):
#   data/eval/e2synth/inputs.jsonl   {id, source, slice, length, synthetic}
#   data/eval/e2synth/labels.jsonl   {id, reference, slice, synthetic}
#   data/eval/e2synth/manifest.json  seed, sha256 of both files, constituent E2 ids per item, ...
#
# Buckets are half-open in raw French characters, [lo, hi): 400-600, 600-800, 800-900; exactly
# `PER_BUCKET` items each. Selection is seeded rejection sampling: draw k in {2,3,4}, draw k
# distinct E2 pairs, order them by E2 id, join, accept if the joined raw French length falls in a
# bucket that still has room and the exact constituent set is new. Item ids are assigned after a
# seeded shuffle so any prefix of the file mixes all three buckets.
#
# CLI: `python -m nmt.data.e2synth [--seed 1234] [--e2-dir data/eval/e2]
#   [--out-dir data/eval/e2synth]`
import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nmt.data.prepare import _sha256_of, _write_json, _write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]

# (slice name, lo inclusive, hi exclusive) in French characters of the raw source text.
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("e2synth_400_600", 400, 600),
    ("e2synth_600_800", 600, 800),
    ("e2synth_800_900", 800, 900),
)
BUCKET_NAMES: tuple[str, ...] = tuple(b[0] for b in BUCKETS)
PER_BUCKET = 100
MIN_PAIRS, MAX_PAIRS = 2, 4
MAX_ATTEMPTS = 2_000_000  # safety stop for the rejection sampler; real runs need well under 1e5
LABEL = "E2-synth (synthetic)"

REUSE_NOTE = (
    "E2-synth is SYNTHETIC: every item concatenates 2-4 sentences of E2, the decoding-tuning "
    "set. It therefore reuses the very sentences the decoding configuration was "
    "tuned on and measures length generalization, not held-out content. Evaluation/reporting "
    "only; never used for checkpoint or decoding selection."
)


def bucket_of(length: int) -> str | None:
    """Name of the bucket whose half-open range [lo, hi) contains `length`, else None."""
    for name, lo, hi in BUCKETS:
        if lo <= length < hi:
            return name
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_items(
    e2_inputs: Sequence[dict[str, Any]],
    e2_labels: Sequence[dict[str, Any]],
    seed: int = 1234,
    per_bucket: int = PER_BUCKET,
) -> list[dict[str, Any]]:
    """Build the E2-synth items from E2 rows. Deterministic given `seed` and the E2 rows.

    Returns dicts `{id, source, reference, slice, length, e2_ids}` in final (shuffled) order,
    ids `e2synth_00000`... Raises `RuntimeError` if the buckets cannot be filled.
    """
    ref_by_id = {r["id"]: r["reference"] for r in e2_labels}
    pairs = [(r["id"], r["source"], ref_by_id[r["id"]]) for r in e2_inputs]
    if any(not src.strip() or not ref.strip() for _, src, ref in pairs):
        raise ValueError("E2 contains an empty source or reference")
    pairs.sort(key=lambda p: p[0])  # fixed base order, independent of file order quirks

    rng = random.Random(seed)
    chosen: dict[str, list[dict[str, Any]]] = {name: [] for name in BUCKET_NAMES}
    seen_sets: set[tuple[str, ...]] = set()
    for _ in range(MAX_ATTEMPTS):
        if all(len(v) >= per_bucket for v in chosen.values()):
            break
        k = rng.randint(MIN_PAIRS, MAX_PAIRS)
        picks = sorted(rng.sample(range(len(pairs)), k))
        ids = tuple(pairs[i][0] for i in picks)
        source = " ".join(pairs[i][1] for i in picks)
        name = bucket_of(len(source))
        if name is None or len(chosen[name]) >= per_bucket or ids in seen_sets:
            continue
        seen_sets.add(ids)
        chosen[name].append(
            {
                "source": source,
                "reference": " ".join(pairs[i][2] for i in picks),
                "slice": name,
                "length": len(source),
                "e2_ids": list(ids),
            }
        )
    else:
        counts = {k: len(v) for k, v in chosen.items()}
        raise RuntimeError(f"could not fill every bucket in {MAX_ATTEMPTS} attempts: {counts}")

    items = [item for name in BUCKET_NAMES for item in chosen[name]]
    rng.shuffle(items)
    for i, item in enumerate(items):
        item["id"] = f"e2synth_{i:05d}"
    return items


def write_e2synth(
    e2_dir: Path = REPO_ROOT / "data" / "eval" / "e2",
    out_dir: Path = REPO_ROOT / "data" / "eval" / "e2synth",
    seed: int = 1234,
    per_bucket: int = PER_BUCKET,
) -> dict[str, Any]:
    """Build E2-synth from `e2_dir` and write inputs.jsonl, labels.jsonl and manifest.json to
    `out_dir`; returns the manifest dict. `per_bucket` is 100 for the real set (tests shrink it)."""
    items = build_items(
        _read_jsonl(e2_dir / "inputs.jsonl"), _read_jsonl(e2_dir / "labels.jsonl"), seed, per_bucket
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path, labels_path = out_dir / "inputs.jsonl", out_dir / "labels.jsonl"
    _write_jsonl(
        inputs_path,
        [
            {
                "id": it["id"],
                "source": it["source"],
                "slice": it["slice"],
                "length": it["length"],
                "synthetic": True,
            }
            for it in items
        ],
    )
    _write_jsonl(
        labels_path,
        [
            {"id": it["id"], "reference": it["reference"], "slice": it["slice"], "synthetic": True}
            for it in items
        ],
    )
    manifest: dict[str, Any] = {
        "name": "e2synth",
        "label": LABEL,
        "synthetic": True,
        "seed": seed,
        "note": REUSE_NOTE,
        "built_from": "data/eval/e2/{inputs,labels}.jsonl",
        "construction": (
            f"seeded rejection sampling: k in [{MIN_PAIRS}, {MAX_PAIRS}] distinct E2 pairs, "
            "ordered by E2 id, fr and en each joined with a single space; accepted if the raw "
            "French length falls in a bucket with room and the constituent set is new; items "
            "then shuffled (seeded) and numbered"
        ),
        "buckets_fr_chars_half_open": {name: [lo, hi] for name, lo, hi in BUCKETS},
        "per_bucket": per_bucket,
        "bucket_counts": {name: sum(it["slice"] == name for it in items) for name in BUCKET_NAMES},
        "n_items": len(items),
        "files_sha256": {
            "inputs.jsonl": _sha256_of(inputs_path),
            "labels.jsonl": _sha256_of(labels_path),
        },
        "items": [
            {"id": it["id"], "slice": it["slice"], "length": it["length"], "e2_ids": it["e2_ids"]}
            for it in items
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the E2-synth long-input probe.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--e2-dir", type=Path, default=REPO_ROOT / "data" / "eval" / "e2")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "eval" / "e2synth")
    args = parser.parse_args(argv)
    m = write_e2synth(args.e2_dir, args.out_dir, args.seed)
    print(
        f"e2synth: {m['n_items']} items, buckets={m['bucket_counts']}, "
        f"inputs sha256={m['files_sha256']['inputs.jsonl']}, "
        f"labels sha256={m['files_sha256']['labels.jsonl']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
