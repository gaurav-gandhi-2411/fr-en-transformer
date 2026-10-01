from __future__ import annotations

# Data pipeline: download opus-100 (en-fr) and opus_books (en-fr), normalize, filter,
# dedupe, build the held-out eval proxies E1/E2/E3, run the leakage guard against
# dev/test/E1/E3, and hold out E2. Writes `train.jsonl`, `data/eval/{e1,e2,e3}/*.jsonl`
# and `data_manifest.json`. Spec §3.
#
# Ordering (binding, see PLAN.md and spec §3): the protected set P (normalized dev
# sources, test sources, dev references) is built first. E1 (opus-100 validation) and
# E3 (opus_books train) are built next, filtered against P only. The train leakage
# guard (step 7) then runs against the union of P, E1 and E3 -- *not* E2, which does
# not exist yet. E2 (long-input proxy) is selected from the post-guard train pairs
# last (step 8) and held out, removing itself and any near-duplicate from train. A
# single `random.Random(seed)` instance is used for both the E3 sample and, later, the
# E2 sample, in that order, so the whole pipeline -- including which rows are sampled
# -- is deterministic given `seed` and the (deterministic) row order coming out of
# each filter step.
#
# The core pipeline (`build_pipeline`) takes plain in-memory iterables and does not
# call the network at call time, so it is exercised in tests without network access.
# `load_raw_datasets` is the thin, network-touching wrapper around it, and `prepare`
# is the top-level entry point used by the CLI.
import argparse
import hashlib
import json
import logging
import platform
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import HfApi

from nmt.data.leakage import MatchIndex, check_overlap, pair_hits, scan_and_filter
from nmt.data.normalize import normalize_text

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# opus-100's `test` split is never requested (spec §3). This is the code-level guard:
# any call site that ever passed "test" here would raise immediately. Caveat, also
# recorded in the manifest: the `datasets` library's opus-100 loading script is an
# old-style builder that regenerates all three splits' local Arrow cache files as a
# side effect of loading any *one* of them -- that is a `datasets`-internal disk-cache
# detail we do not control, not a read of the test split's contents. This code never
# requests, iterates, or otherwise touches the test split's rows.
_OPUS100_ALLOWED_SPLITS = frozenset({"train", "validation"})

NORMALIZATION_DEFINITION = (
    'NFKC normalization; apostrophe variants (’ ‘ ʼ) unified to "\'"; quote '
    "variants (« » “ ”) unified to '\"'; runs of whitespace collapsed to a "
    "single space and the result stripped; casing preserved. "
    "See nmt/data/normalize.py:normalize_text."
)

LENGTH_CAP_NOTE = (
    "The 256-subword-token-per-side length cap (spec §3) is applied in P2 tokenize.py, not here; "
    "train.jsonl rows produced by this module are unbounded by subword length."
)

OPUS100_TEST_SPLIT_NOTE = (
    "The opus-100 `test` split is never requested by this code (only 'train'/'validation' are "
    "passed to datasets.load_dataset; see _OPUS100_ALLOWED_SPLITS). Note that the `datasets` "
    "library's opus-100 loading script regenerates all three splits' local Arrow cache files as "
    "a side effect of loading any single split -- this code never reads or uses the test split's "
    "rows."
)


# --------------------------------------------------------------------------------------
# Core, network-free pipeline
# --------------------------------------------------------------------------------------


@dataclass
class PipelineOutput:
    """Result of `build_pipeline`: rows ready to write, plus everything needed for the
    manifest and the post-hoc `check_overlap` verification."""

    train_rows: list[dict[str, str]]
    eval_rows: dict[str, list[dict[str, Any]]]
    protected_strings: dict[str, list[str]]
    stats: dict[str, Any]


def _to_rows(pairs: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Attach normalized fr/en fields to raw (fr, en) pairs, keeping the raw text too
    (eval sets are written with raw, un-normalized text; spec §3)."""
    rows = []
    for p in pairs:
        fr_raw, en_raw = p["fr"], p["en"]
        rows.append(
            {
                "fr_raw": fr_raw,
                "en_raw": en_raw,
                "fr": normalize_text(fr_raw),
                "en": normalize_text(en_raw),
            }
        )
    return rows


def _drop_empty_side(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop pairs where either normalized side is empty (step 2)."""
    before = len(rows)
    kept = [r for r in rows if r["fr"] and r["en"]]
    return kept, before - len(kept)


def _drop_exact_duplicates(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop exact duplicate (fr, en) pairs, keeping the first occurrence (step 3)."""
    seen: set[tuple[str, str]] = set()
    kept = []
    for r in rows:
        key = (r["fr"], r["en"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(rows) - len(kept)


def _drop_fr_eq_en(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop pairs where fr == en after normalization (step 4)."""
    before = len(rows)
    kept = [r for r in rows if r["fr"] != r["en"]]
    return kept, before - len(kept)


def _length_ratio_ok(fr: str, en: str) -> bool:
    """True iff the char length ratio len(fr)/len(en) falls within [1/3, 3]."""
    lf, le = len(fr), len(en)
    if lf == 0 or le == 0:
        return False  # defensive; empty sides are already dropped by this point
    ratio = lf / le
    return (1.0 / 3.0) <= ratio <= 3.0


def _drop_length_ratio(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop pairs whose char length ratio falls outside [1/3, 3] (step 5)."""
    before = len(rows)
    kept = [r for r in rows if _length_ratio_ok(r["fr"], r["en"])]
    return kept, before - len(kept)


def _non_letter_fraction(s: str) -> float:
    """Fraction of non-letter characters in `s`, counted over all characters excluding
    whitespace. Letter = `str.isalpha()`. An all-whitespace string returns 1.0
    (defensively treated as 100% non-letter; empty sides are already dropped by the
    time this runs, so this only guards against a whitespace-only side slipping through)."""
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 1.0
    non_letter = sum(1 for c in chars if not c.isalpha())
    return non_letter / len(chars)


def _drop_high_non_letter(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop pairs where either side has >50% non-letter characters (step 6)."""
    before = len(rows)
    kept = [
        r
        for r in rows
        if _non_letter_fraction(r["fr"]) <= 0.5 and _non_letter_fraction(r["en"]) <= 0.5
    ]
    return kept, before - len(kept)


def _build_eval_candidates(
    pairs: Iterable[Mapping[str, str]], p_index: MatchIndex
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Normalize `pairs`, drop empty sides, then drop any pair matching the protected
    index `p_index` (exact or near-dup, either side). Used identically for E1 and E3
    (both are filtered against P only, spec §3)."""
    rows = _to_rows(pairs)
    raw_count = len(rows)
    rows = [r for r in rows if r["fr"] and r["en"]]
    dropped_empty = raw_count - len(rows)
    kept, leak_stats = scan_and_filter(rows, "fr", "en", {"p": p_index})
    stats = {
        "raw": raw_count,
        "dropped_empty": dropped_empty,
        "dropped_leak_exact": leak_stats["exact_hits"],
        "dropped_leak_near_dup": leak_stats["near_dup_only_hits"],
        "kept": len(kept),
    }
    return kept, stats


def build_pipeline(
    train_pairs: Sequence[Mapping[str, str]],
    val_pairs: Sequence[Mapping[str, str]],
    books_pairs: Sequence[Mapping[str, str]],
    dev_sources: Sequence[str],
    test_sources: Sequence[str],
    dev_references: Sequence[str],
    seed: int,
) -> PipelineOutput:
    """Core, network-free data pipeline (spec §3; ordering documented in the module
    docstring). `train_pairs`/`val_pairs` are opus-100 (en-fr) train/validation rows;
    `books_pairs` are opus_books (en-fr) train rows. Each pair is a mapping with raw
    "fr"/"en" text. `dev_sources`/`test_sources`/`dev_references` are the raw text of
    the provided dev/test files.
    """
    rng = random.Random(seed)
    filters: list[dict[str, Any]] = []

    def _log(step: str, removed: int, remaining: int) -> None:
        filters.append({"step": step, "removed": removed, "remaining": remaining})
        logger.info("filter step=%s removed=%d remaining=%d", step, removed, remaining)

    # --- P: the protected set (dev/test sources, dev references), normalized. ---
    dev_src_norm = [normalize_text(s) for s in dev_sources]
    test_src_norm = [normalize_text(s) for s in test_sources]
    dev_ref_norm = [normalize_text(s) for s in dev_references]
    dev_src_index = MatchIndex.from_strings(dev_src_norm)
    test_src_index = MatchIndex.from_strings(test_src_norm)
    dev_ref_index = MatchIndex.from_strings(dev_ref_norm)
    p_index = dev_src_index | test_src_index | dev_ref_index

    # --- TRAIN steps 1-6 ---
    rows = _to_rows(train_pairs)
    _log("normalize", 0, len(rows))
    rows, removed = _drop_empty_side(rows)
    _log("drop_empty_side", removed, len(rows))
    rows, removed = _drop_exact_duplicates(rows)
    _log("drop_exact_duplicate_pairs", removed, len(rows))
    rows, removed = _drop_fr_eq_en(rows)
    _log("drop_fr_eq_en", removed, len(rows))
    rows, removed = _drop_length_ratio(rows)
    _log("drop_length_ratio_out_of_bounds", removed, len(rows))
    rows, removed = _drop_high_non_letter(rows)
    _log("drop_high_non_letter_fraction", removed, len(rows))

    # --- E1 (opus-100 validation) and E3 (opus_books train): built vs P only, BEFORE
    # the train leakage guard (spec §3 ordering note). rng call #1 is E3's sample. ---
    e1_rows, e1_stats = _build_eval_candidates(val_pairs, p_index)
    e1_stats["sampled"] = len(e1_rows)  # E1 is not subsampled: all survivors are kept

    e3_candidates, e3_stats = _build_eval_candidates(books_pairs, p_index)
    e3_sample_size = min(1000, len(e3_candidates))
    e3_rows = rng.sample(e3_candidates, k=e3_sample_size)  # rng call #1
    e3_stats["sampled"] = e3_sample_size

    # --- step 7: train leakage guard vs union(dev_src, test_src, dev_ref, E1, E3) ---
    e1_index = MatchIndex.from_strings([r["fr"] for r in e1_rows] + [r["en"] for r in e1_rows])
    e3_index = MatchIndex.from_strings([r["fr"] for r in e3_rows] + [r["en"] for r in e3_rows])
    guard_indices = {
        "dev_src": dev_src_index,
        "test_src": test_src_index,
        "dev_ref": dev_ref_index,
        "e1": e1_index,
        "e3": e3_index,
    }
    rows, guard_stats = scan_and_filter(rows, "fr", "en", guard_indices)
    _log("leakage_guard_vs_dev_test_e1_e3", guard_stats["total_removed"], len(rows))

    # --- step 8: E2 (long-input proxy), held out from the post-guard train pairs.
    # rng call #2 is E2's sample -- always after E3's, per the ordering note. ---
    e2_candidate_pool = [r for r in rows if len(r["fr"]) > 200]
    e2_sample_size = min(1000, len(e2_candidate_pool))
    e2_rows = rng.sample(e2_candidate_pool, k=e2_sample_size)  # rng call #2
    e2_sample_ids = {id(r) for r in e2_rows}
    e2_index = MatchIndex.from_strings([r["fr"] for r in e2_rows] + [r["en"] for r in e2_rows])

    train_final: list[dict[str, str]] = []
    removed_as_holdout = 0
    removed_as_near_dup = 0
    for r in rows:
        if id(r) in e2_sample_ids:
            removed_as_holdout += 1
            continue
        exact_hit, near_dup_hit = pair_hits(r["fr"], r["en"], e2_index)
        if exact_hit or near_dup_hit:
            removed_as_near_dup += 1
            continue
        train_final.append(r)
    _log(
        "e2_holdout_and_near_dup_removal",
        removed_as_holdout + removed_as_near_dup,
        len(train_final),
    )

    # --- assemble outputs ---
    train_out = [
        {"id": f"train_{i:07d}", "fr": r["fr"], "en": r["en"]} for i, r in enumerate(train_final)
    ]

    def _eval_out(prefix: str, rows_: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                "id": f"{prefix}_{i:05d}",
                "source": r["fr_raw"],
                "reference": r["en_raw"],
                "slice": prefix,
                "length": len(r["fr_raw"]),
            }
            for i, r in enumerate(rows_)
        ]

    eval_rows = {
        "e1": _eval_out("e1", e1_rows),
        "e2": _eval_out("e2", e2_rows),
        "e3": _eval_out("e3", e3_rows),
    }

    protected_strings = {
        "dev_src": dev_src_norm,
        "test_src": test_src_norm,
        "dev_ref": dev_ref_norm,
        "e1": [r["fr"] for r in e1_rows] + [r["en"] for r in e1_rows],
        "e2": [r["fr"] for r in e2_rows] + [r["en"] for r in e2_rows],
        "e3": [r["fr"] for r in e3_rows] + [r["en"] for r in e3_rows],
    }

    stats = {
        "filters": filters,
        "leakage_guard": {
            "exact_hits": guard_stats["exact_hits"],
            "near_dup_only_hits": guard_stats["near_dup_only_hits"],
            "per_source_hits": guard_stats["per_source_hits"],
        },
        "e2": {
            "candidates_len_fr_gt_200": len(e2_candidate_pool),
            "sampled": e2_sample_size,
            "removed_as_e2_holdout": removed_as_holdout,
            "removed_as_e2_near_dup_or_duplicate": removed_as_near_dup,
        },
        "eval_sets": {"e1": e1_stats, "e2": {"kept": len(e2_rows)}, "e3": e3_stats},
        "final_train_pair_count": len(train_final),
        "length_cap_note": LENGTH_CAP_NOTE,
    }

    return PipelineOutput(
        train_rows=train_out, eval_rows=eval_rows, protected_strings=protected_strings, stats=stats
    )


# --------------------------------------------------------------------------------------
# Thin, network-touching IO wrapper
# --------------------------------------------------------------------------------------


@dataclass
class RawDatasets:
    train_pairs: list[dict[str, str]]
    val_pairs: list[dict[str, str]]
    books_pairs: list[dict[str, str]]
    dataset_info: dict[str, Any]


def _load_opus100_split(split: str) -> datasets.Dataset:
    """Load one opus-100 (en-fr) split. See `_OPUS100_ALLOWED_SPLITS` and
    `OPUS100_TEST_SPLIT_NOTE` for the test-split guarantee and its caveat."""
    assert split in _OPUS100_ALLOWED_SPLITS, f"refusing to load opus-100 split={split!r}"
    return datasets.load_dataset("Helsinki-NLP/opus-100", "en-fr", split=split)


def load_raw_datasets(limit: int | None) -> RawDatasets:
    """Download opus-100 (en-fr) train+validation and opus_books (en-fr) train, record
    each dataset's revision SHA and raw row counts, and return plain in-memory pairs.
    `limit` subsamples the raw opus-100 *train* split only (fast dev/test iteration;
    `raw_train_count` in the returned `dataset_info` is always the true, un-limited
    size, so `share_of_opus100_train_used` stays meaningful under `--limit`)."""
    api = HfApi()
    opus100_sha = api.dataset_info("Helsinki-NLP/opus-100").sha
    opus_books_sha = api.dataset_info("Helsinki-NLP/opus_books").sha

    train_ds = _load_opus100_split("train")
    val_ds = _load_opus100_split("validation")
    books_ds = datasets.load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")

    raw_train_count = len(train_ds)
    raw_val_count = len(val_ds)
    raw_books_count = len(books_ds)

    if limit is not None:
        train_ds = train_ds.select(range(min(limit, len(train_ds))))

    def _pairs(ds: datasets.Dataset) -> list[dict[str, str]]:
        return [{"fr": row["translation"]["fr"], "en": row["translation"]["en"]} for row in ds]

    dataset_info = {
        "opus100": {
            "repo_id": "Helsinki-NLP/opus-100",
            "config": "en-fr",
            "revision_sha": opus100_sha,
            "raw_train_count": raw_train_count,
            "raw_validation_count": raw_val_count,
        },
        "opus_books": {
            "repo_id": "Helsinki-NLP/opus_books",
            "config": "en-fr",
            "revision_sha": opus_books_sha,
            "raw_train_count": raw_books_count,
        },
    }
    return RawDatasets(
        train_pairs=_pairs(train_ds),
        val_pairs=_pairs(val_ds),
        books_pairs=_pairs(books_ds),
        dataset_info=dataset_info,
    )


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------


def _read_jsonl_field(path: Path, field: str) -> list[str]:
    """Read one string field from every row of a jsonl file, in file order."""
    values: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(json.loads(line)[field])
    return values


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write one JSON object per line, LF line endings (deterministic across OSes)."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_or_str(path: Path, root: Path) -> str:
    """`path` relative to `root` as a posix string, or the plain string if `path`
    isn't under `root` (e.g. a test writing to a tmp_path outside the repo)."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------------------


def prepare(
    out_dir: Path,
    seed: int,
    limit: int | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Load the real datasets, run `build_pipeline`, write `train.jsonl`, the eval
    sets and the manifest to disk, and return the manifest dict (so a future training
    pipeline can log it to W&B without re-reading it from disk; spec §3)."""
    start = time.monotonic()
    out_dir = Path(out_dir)
    eval_dir = repo_root / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for slice_name in ("e1", "e2", "e3"):
        (eval_dir / slice_name).mkdir(parents=True, exist_ok=True)

    raw = load_raw_datasets(limit)

    dev_sources = _read_jsonl_field(repo_root / "data" / "dev" / "inputs.jsonl", "source")
    test_sources = _read_jsonl_field(repo_root / "data" / "test" / "inputs.jsonl", "source")
    dev_references = _read_jsonl_field(repo_root / "data" / "dev" / "labels.jsonl", "reference")

    output = build_pipeline(
        train_pairs=raw.train_pairs,
        val_pairs=raw.val_pairs,
        books_pairs=raw.books_pairs,
        dev_sources=dev_sources,
        test_sources=test_sources,
        dev_references=dev_references,
        seed=seed,
    )

    train_path = out_dir / "train.jsonl"
    _write_jsonl(train_path, output.train_rows)
    output_files: dict[str, str] = {_rel_or_str(train_path, repo_root): _sha256_of(train_path)}

    for slice_name in ("e1", "e2", "e3"):
        rows = output.eval_rows[slice_name]
        inputs_path = eval_dir / slice_name / "inputs.jsonl"
        labels_path = eval_dir / slice_name / "labels.jsonl"
        _write_jsonl(
            inputs_path,
            [
                {"id": r["id"], "source": r["source"], "slice": r["slice"], "length": r["length"]}
                for r in rows
            ],
        )
        _write_jsonl(
            labels_path,
            [{"id": r["id"], "reference": r["reference"], "slice": r["slice"]} for r in rows],
        )
        output_files[_rel_or_str(inputs_path, repo_root)] = _sha256_of(inputs_path)
        output_files[_rel_or_str(labels_path, repo_root)] = _sha256_of(labels_path)

    # Independently re-read train.jsonl from disk and verify zero overlap (spec §12).
    post_check = check_overlap(train_path, output.protected_strings)

    raw_opus100_train_count = raw.dataset_info["opus100"]["raw_train_count"]
    share_used = (
        output.stats["final_train_pair_count"] / raw_opus100_train_count
        if raw_opus100_train_count
        else 0.0
    )

    manifest: dict[str, Any] = {
        "seed": seed,
        "limit": limit,
        "created_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "datasets_version": datasets.__version__,
        "wall_clock_seconds": round(time.monotonic() - start, 3),
        "datasets": raw.dataset_info,
        "normalization": NORMALIZATION_DEFINITION,
        "opus100_test_split_note": OPUS100_TEST_SPLIT_NOTE,
        **output.stats,
        "share_of_opus100_train_used": share_used,
        "output_files_sha256": output_files,
        "post_check": post_check,
    }

    _write_json(out_dir / "data_manifest.json", manifest)
    _write_json(repo_root / "data" / "data_manifest.json", manifest)

    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fr-en-transformer data pipeline: prepare train.jsonl, "
        "the E1/E2/E3 eval proxies and data_manifest.json (spec §3)."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed"), help="Output dir for train.jsonl"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for dedup/sampling")
    parser.add_argument(
        "--limit", type=int, default=None, help="Subsample raw opus-100 train (dev/test iteration)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    manifest = prepare(out_dir=args.out, seed=args.seed, limit=args.limit)
    logger.info(
        "prepare complete: final_train_pair_count=%d share_of_opus100_train_used=%.4f "
        "wall_clock_seconds=%.1f",
        manifest["final_train_pair_count"],
        manifest["share_of_opus100_train_used"],
        manifest["wall_clock_seconds"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
