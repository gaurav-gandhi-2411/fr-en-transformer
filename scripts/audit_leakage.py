from __future__ import annotations

# Read-only leakage audit. Re-derives, by running `nmt.data.prepare.build_pipeline`
# itself (datasets read from the local HF cache at the revisions pinned in data_manifest.json,
# never the network), the exact pool of train pairs entering the step-7 leakage guard and the
# exact pairs the guard removes, then explains *what* was removed: by matched target set, by
# exact vs near-duplicate match, by which side of the train pair matched, and by the train
# pair's French word count. It also explains E1's 2000 -> 1940 drop and E2's 1001 removals.
#
# Nothing here changes the leakage rule or any data artifact: `prepare.scan_and_filter` is wrapped
# only to *capture* the guard's input rows/indices/output (the wrapper calls the real function and
# returns its result unchanged), and the only files written are
# reports/audit/leakage_audit.{json,md}.
# Normalization, the near-dup key and the match primitives are reused from nmt.data
# (`normalize_text`, `near_dup_key`, `MatchIndex`, `pair_hits`), never re-implemented.
#
# Every headline total is asserted against data/data_manifest.json; a mismatch fails loudly.
#
# CLI: `uv run python scripts/audit_leakage.py [--out-dir reports/audit] [--seed 1234]`
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import datasets

from nmt.data import prepare
from nmt.data.leakage import MatchIndex, pair_hits
from nmt.data.normalize import near_dup_key

REPO_ROOT = Path(__file__).resolve().parents[1]

# The five indices the real guard runs against (prepare.build_pipeline step 7). E2 is NOT among
# them: E2 is carved out of the post-guard pool afterwards (step 8), see section (f).
FINE_SETS: tuple[str, ...] = ("dev_src", "dev_ref", "test_src", "e1", "e3")
# Coarse target sets the audit reports on. dev = dev_src U dev_ref. e2 is structurally 0 for the
# guard step (it does not exist yet) and is listed so the table is complete.
COARSE_SETS: tuple[str, ...] = ("dev", "test", "e1", "e2", "e3")
COARSE_MEMBERS: dict[str, tuple[str, ...]] = {
    "dev": ("dev_src", "dev_ref"),
    "test": ("test_src",),
    "e1": ("e1",),
    "e2": (),
    "e3": ("e3",),
}
# Exclusive-by-first-match priority: official sets before proxies, E1 before E3 (E2 never matches
# in the guard step). A pair matching several sets is attributed to the first one listed here.
PRIORITY: tuple[str, ...] = ("dev", "test", "e1", "e2", "e3")

WORD_BUCKETS: tuple[str, ...] = ("1-3", "4-10", ">10")
MATCH_TYPES: tuple[str, ...] = ("exact", "near_dup")
SIDES: tuple[str, ...] = ("fr", "en", "both")


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested; no data loading)
# --------------------------------------------------------------------------------------


def word_bucket(n_words: int) -> str:
    """French source word-count bucket used in the audit: 1-3, 4-10, >10."""
    if n_words <= 3:
        return "1-3"
    if n_words <= 10:
        return "4-10"
    return ">10"


def fr_word_count(fr_normalized: str) -> int:
    """Whitespace-split word count of an already-normalized French string."""
    return len(fr_normalized.split())


def side_label(fr_hit: bool, en_hit: bool) -> str | None:
    """'fr' / 'en' / 'both' for which side(s) of the train pair matched, None if neither."""
    if fr_hit and en_hit:
        return "both"
    if fr_hit:
        return "fr"
    if en_hit:
        return "en"
    return None


def merge_set_matches(parts: Sequence[Mapping[str, str] | None]) -> dict[str, str] | None:
    """Merge several fine-set matches (each `{"match_type", "side"}` or None) into one coarse-set
    match: exact if any part is exact, side = union of the parts' sides."""
    present = [p for p in parts if p is not None]
    if not present:
        return None
    match_type = "exact" if any(p["match_type"] == "exact" for p in present) else "near_dup"
    sides: set[str] = set()
    for p in present:
        sides |= {"fr", "en"} if p["side"] == "both" else {p["side"]}
    return {"match_type": match_type, "side": side_label("fr" in sides, "en" in sides) or "fr"}


def coarse_matches(fine: Mapping[str, Mapping[str, str] | None]) -> dict[str, dict[str, str]]:
    """Group fine-set matches (dev_src/dev_ref/test_src/e1/e3) into the coarse sets; only sets
    that actually matched appear in the result."""
    out: dict[str, dict[str, str]] = {}
    for name, members in COARSE_MEMBERS.items():
        merged = merge_set_matches([fine.get(m) for m in members])
        if merged is not None:
            out[name] = merged
    return out


def first_match(matched: Sequence[str], priority: Sequence[str] = PRIORITY) -> str:
    """The highest-priority set among `matched` (raises if `matched` has no known set)."""
    for name in priority:
        if name in matched:
            return name
    raise ValueError(f"no known set in {list(matched)!r}")


def _empty_side_counter() -> dict[str, int]:
    return dict.fromkeys(SIDES, 0)


def breakdown_non_exclusive(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A pair counts in every coarse set it matches. Per set: total, by match type, by side, and
    match type x side. Each record needs `"sets"`: {coarse set: {"match_type", "side"}}."""
    out: dict[str, Any] = {}
    for name in COARSE_SETS:
        by_type = dict.fromkeys(MATCH_TYPES, 0)
        by_side = _empty_side_counter()
        by_type_side = {t: _empty_side_counter() for t in MATCH_TYPES}
        total = 0
        for r in records:
            m = r["sets"].get(name)
            if m is None:
                continue
            total += 1
            by_type[m["match_type"]] += 1
            by_side[m["side"]] += 1
            by_type_side[m["match_type"]][m["side"]] += 1
        out[name] = {
            "total": total,
            "by_match_type": by_type,
            "by_side": by_side,
            "by_match_type_and_side": by_type_side,
        }
    return out


def breakdown_exclusive(
    records: Sequence[Mapping[str, Any]], priority: Sequence[str] = PRIORITY
) -> dict[str, Any]:
    """Each pair is attributed to exactly one set: the first of `priority` it matches. The match
    type is the PAIR-level one (exact if it is exact against any set, as in the manifest's
    exact/near_dup_only split), so the table sums to the manifest totals; the side is the
    attributed set's own side. Totals over all sets equal len(records)."""
    out: dict[str, Any] = {}
    for name in COARSE_SETS:
        out[name] = {
            "total": 0,
            "by_match_type": dict.fromkeys(MATCH_TYPES, 0),
            "by_side": _empty_side_counter(),
            "by_match_type_and_side": {t: _empty_side_counter() for t in MATCH_TYPES},
        }
    for r in records:
        name = first_match(list(r["sets"]), priority)
        side = r["sets"][name]["side"]
        cell = out[name]
        cell["total"] += 1
        cell["by_match_type"][r["match_type"]] += 1
        cell["by_side"][side] += 1
        cell["by_match_type_and_side"][r["match_type"]][side] += 1
    return out


def breakdown_word_buckets(
    records: Sequence[Mapping[str, Any]], pool_bucket_counts: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Removals by French word-count bucket crossed with pair-level match type, the share of
    near-dup-only removals that are 1-3-word sentences, and (if given) each bucket's removal
    rate relative to the guard's input pool."""
    cross = {b: dict.fromkeys(MATCH_TYPES, 0) for b in WORD_BUCKETS}
    for r in records:
        cross[r["bucket"]][r["match_type"]] += 1
    n_near = sum(cross[b]["near_dup"] for b in WORD_BUCKETS)
    n_exact = sum(cross[b]["exact"] for b in WORD_BUCKETS)
    n_all = n_near + n_exact
    out: dict[str, Any] = {
        "by_bucket_and_match_type": {
            b: {**cross[b], "total": cross[b]["exact"] + cross[b]["near_dup"]} for b in WORD_BUCKETS
        },
        "share_of_near_dup_removals_that_are_1_3_words": (
            cross["1-3"]["near_dup"] / n_near if n_near else 0.0
        ),
        "share_of_all_removals_that_are_1_3_words": (
            (cross["1-3"]["near_dup"] + cross["1-3"]["exact"]) / n_all if n_all else 0.0
        ),
    }
    if pool_bucket_counts is not None:
        pool_total = sum(pool_bucket_counts.values())
        out["pool_bucket_counts"] = dict(pool_bucket_counts)
        out["pool_bucket_share"] = {
            b: pool_bucket_counts[b] / pool_total if pool_total else 0.0 for b in WORD_BUCKETS
        }
        out["removal_rate_per_bucket"] = {
            b: (cross[b]["exact"] + cross[b]["near_dup"]) / pool_bucket_counts[b]
            if pool_bucket_counts[b]
            else 0.0
            for b in WORD_BUCKETS
        }
    return out


# --------------------------------------------------------------------------------------
# Matching detail (uses the pipeline's own MatchIndex / pair_hits)
# --------------------------------------------------------------------------------------


class SetMatcher:
    """One fine target set: its `MatchIndex` plus a near-key -> original strings map, so a match
    can be explained with the actual target string(s) it hit."""

    def __init__(self, strings: Sequence[str]) -> None:
        self.index = MatchIndex.from_strings(strings)
        self._by_key: dict[str, list[str]] = defaultdict(list)
        for s in dict.fromkeys(strings):  # de-duplicated, order-preserving
            k = near_dup_key(s)
            if s and k:
                self._by_key[k].append(s)

    def match(self, fr: str, en: str) -> dict[str, str] | None:
        """`{"match_type", "side"}` for a (fr, en) pair against this set, or None."""
        fr_exact, fr_near = self.index.hit(fr)
        en_exact, en_near = self.index.hit(en)
        side = side_label(fr_exact or fr_near, en_exact or en_near)
        if side is None:
            return None
        return {"match_type": "exact" if (fr_exact or en_exact) else "near_dup", "side": side}

    def targets_for(self, s: str, limit: int = 3) -> list[str]:
        """The target strings in this set that `s` hit (exact string and/or same near-dup key)."""
        found: list[str] = []
        if s in self.index.exact:
            found.append(s)
        for t in self._by_key.get(near_dup_key(s), []):
            if t not in found:
                found.append(t)
        return found[:limit]


def fine_matches(
    fr: str, en: str, matchers: Mapping[str, SetMatcher]
) -> dict[str, dict[str, str] | None]:
    return {name: m.match(fr, en) for name, m in matchers.items()}


def describe_targets(
    fr: str, en: str, matchers: Mapping[str, SetMatcher], fine: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """For each fine set matched, the train side(s) that matched and the target string(s) hit."""
    out = []
    for name, m in fine.items():
        if m is None:
            continue
        entry: dict[str, Any] = {"set": name, **m, "targets": {}}
        for side, text in (("fr", fr), ("en", en)):
            hits = matchers[name].targets_for(text)
            if hits:
                entry["targets"][side] = hits
        out.append(entry)
    return out


# --------------------------------------------------------------------------------------
# Loading + capture of the real guard step
# --------------------------------------------------------------------------------------


def _read_field(path: Path, field: str) -> list[str]:
    return prepare._read_jsonl_field(path, field)


def load_cached_datasets(manifest: Mapping[str, Any]) -> prepare.RawDatasets:
    """Load opus-100 (en-fr) train/validation and opus_books (en-fr) from the local HF cache only,
    checking that the cache directories carry the revisions pinned in the manifest and that raw
    row counts equal the manifest's. Never touches the network."""
    datasets.config.HF_DATASETS_OFFLINE = True
    info = manifest["datasets"]
    cache = Path(datasets.config.HF_DATASETS_CACHE)
    for key, repo in (
        ("opus100", "Helsinki-NLP___opus-100"),
        ("opus_books", "Helsinki-NLP___opus_books"),
    ):
        sha = info[key]["revision_sha"]
        d = cache / repo / info[key]["config"] / "0.0.0" / sha
        if not d.is_dir():
            raise SystemExit(f"cache for pinned revision missing: {d}")
    train_ds = prepare._load_opus100_split("train")
    val_ds = prepare._load_opus100_split("validation")
    books_ds = datasets.load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")
    counts = (len(train_ds), len(val_ds), len(books_ds))
    expected = (
        info["opus100"]["raw_train_count"],
        info["opus100"]["raw_validation_count"],
        info["opus_books"]["raw_train_count"],
    )
    if counts != expected:
        raise SystemExit(f"raw row counts {counts} != manifest {expected}")

    def _pairs(ds: datasets.Dataset) -> list[dict[str, str]]:
        return [{"fr": row["translation"]["fr"], "en": row["translation"]["en"]} for row in ds]

    return prepare.RawDatasets(_pairs(train_ds), _pairs(val_ds), _pairs(books_ds), dict(info))


def run_pipeline_with_capture(
    raw: prepare.RawDatasets, repo_root: Path, seed: int
) -> tuple[prepare.PipelineOutput, dict[str, Any]]:
    """Run the real `build_pipeline`; capture the guard step's input rows, output rows and
    indices by wrapping `prepare.scan_and_filter` (the wrapper returns the real result as-is)."""
    captured: dict[str, Any] = {}
    real = prepare.scan_and_filter

    def _capturing(
        rows: list[dict], fr_key: str, en_key: str, indices: Mapping[str, MatchIndex]
    ) -> tuple[list[dict], dict]:
        kept, stats = real(rows, fr_key, en_key, indices)
        if set(indices) == set(FINE_SETS):
            if "rows" in captured:
                raise RuntimeError("guard step captured twice")
            captured.update(rows=rows, kept=kept, stats=stats)
        return kept, stats

    prepare.scan_and_filter = _capturing  # type: ignore[assignment]
    try:
        output = prepare.build_pipeline(
            train_pairs=raw.train_pairs,
            val_pairs=raw.val_pairs,
            books_pairs=raw.books_pairs,
            dev_sources=_read_field(repo_root / "data" / "dev" / "inputs.jsonl", "source"),
            test_sources=_read_field(repo_root / "data" / "test" / "inputs.jsonl", "source"),
            dev_references=_read_field(repo_root / "data" / "dev" / "labels.jsonl", "reference"),
            seed=seed,
        )
    finally:
        prepare.scan_and_filter = real  # type: ignore[assignment]
    if "rows" not in captured:
        raise RuntimeError("leakage guard step was never captured")
    return output, captured


# --------------------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------------------


def _example_view(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train_fr": rec["fr"],
        "train_en": rec["en"],
        "word_count": rec["words"],
        "match_type": rec["match_type"],
        "matched_sets": list(rec["sets"]),
        "matched": rec["matched_detail"],
    }


def audit_guard(
    captured: Mapping[str, Any],
    output: prepare.PipelineOutput,
    seed: int,
    n_examples: int = 20,
) -> dict[str, Any]:
    """Sections (a)-(d): classify every removed pair of the guard step."""
    prot = output.protected_strings
    matchers = {
        "dev_src": SetMatcher(prot["dev_src"]),
        "dev_ref": SetMatcher(prot["dev_ref"]),
        "test_src": SetMatcher(prot["test_src"]),
        "e1": SetMatcher(prot["e1"]),
        "e3": SetMatcher(prot["e3"]),
    }
    union = matchers["dev_src"].index
    for name in FINE_SETS[1:]:
        union = union | matchers[name].index

    kept_ids = {id(r) for r in captured["kept"]}
    pool: list[dict[str, str]] = captured["rows"]
    pool_bucket_counts = Counter(word_bucket(fr_word_count(r["fr"])) for r in pool)

    records: list[dict[str, Any]] = []
    for pos, r in enumerate(pool):
        if id(r) in kept_ids:
            continue
        fr, en = r["fr"], r["en"]
        exact_hit, near_hit = pair_hits(fr, en, union)
        if not (exact_hit or near_hit):
            raise RuntimeError(f"pool row {pos} removed by guard but no union hit on re-derive")
        fine = fine_matches(fr, en, matchers)
        words = fr_word_count(fr)
        records.append(
            {
                "pool_pos": pos,
                "fr": fr,
                "en": en,
                "words": words,
                "bucket": word_bucket(words),
                "match_type": "exact" if exact_hit else "near_dup",
                "fine": fine,
                "sets": coarse_matches(fine),
                "matched_detail": describe_targets(fr, en, matchers, fine),
            }
        )

    n_exact = sum(r["match_type"] == "exact" for r in records)
    n_near = sum(r["match_type"] == "near_dup" for r in records)
    per_source = {name: sum(r["fine"][name] is not None for r in records) for name in FINE_SETS}

    totals = {
        "pool_size_entering_guard": len(pool),
        "removed": len(records),
        "exact": n_exact,
        "near_dup_only": n_near,
        "per_source_hits": per_source,
        "per_source_hits_note": (
            "non-exclusive: a removed pair increments every source it hits (as in the manifest)"
        ),
    }

    near_records = [r for r in records if r["match_type"] == "near_dup"]
    sample = random.Random(seed).sample(near_records, k=min(n_examples, len(near_records)))
    return {
        "totals": totals,
        "by_target_set_non_exclusive": breakdown_non_exclusive(records),
        "by_target_set_exclusive": {
            "priority": list(PRIORITY),
            "note": (
                "pair attributed to the first matching set in `priority`; match_type is the "
                "pair-level exact/near_dup (sums to the manifest totals); side is the attributed "
                "set's own side. e2 is 0 by construction: E2 is not part of the step-7 guard."
            ),
            **breakdown_exclusive(records),
        },
        "side_note": (
            "side = which side(s) of the TRAIN pair matched the target set (exact or near-dup); "
            "'both' = fr and en each matched (possibly different target strings)."
        ),
        "by_french_word_count": breakdown_word_buckets(records, dict(pool_bucket_counts)),
        "near_dup_examples": {
            "seed": seed,
            "n": len(sample),
            "population": "near_dup_only removals (the 2496-type), in pool order, random.sample",
            "examples": [_example_view(r) for r in sample],
        },
    }


def audit_e1(output: prepare.PipelineOutput, raw: prepare.RawDatasets, seed: int) -> dict[str, Any]:
    """Section (e): which official set each dropped opus-100 validation row matched. Mirrors
    `prepare._build_eval_candidates`: normalize, drop empty sides, drop against P only."""
    prot = output.protected_strings
    p_matchers = {
        "dev_src": SetMatcher(prot["dev_src"]),
        "dev_ref": SetMatcher(prot["dev_ref"]),
        "test_src": SetMatcher(prot["test_src"]),
    }
    p_union = (
        p_matchers["dev_src"].index | p_matchers["dev_ref"].index | p_matchers["test_src"].index
    )
    rows = prepare._to_rows(raw.val_pairs)
    dropped: list[dict[str, Any]] = []
    for pos, r in enumerate(rows):
        if not (r["fr"] and r["en"]):
            continue
        exact_hit, near_hit = pair_hits(r["fr"], r["en"], p_union)
        if not (exact_hit or near_hit):
            continue
        fine = fine_matches(r["fr"], r["en"], p_matchers)
        dropped.append(
            {
                "val_pos": pos,
                "fr": r["fr"],
                "en": r["en"],
                "match_type": "exact" if exact_hit else "near_dup",
                "sets": fine,
                "matched": describe_targets(r["fr"], r["en"], p_matchers, fine),
            }
        )
    by_type = Counter(d["match_type"] for d in dropped)
    by_set = {
        name: {
            "total": sum(d["sets"][name] is not None for d in dropped),
            "exact": sum(
                d["sets"][name] is not None and d["sets"][name]["match_type"] == "exact"
                for d in dropped
            ),
            "near_dup": sum(
                d["sets"][name] is not None and d["sets"][name]["match_type"] == "near_dup"
                for d in dropped
            ),
        }
        for name in p_matchers
    }
    combos = Counter("+".join(n for n in p_matchers if d["sets"][n] is not None) for d in dropped)
    sample = random.Random(seed).sample(dropped, k=min(5, len(dropped)))
    return {
        "raw_validation_rows": len(rows),
        "dropped": len(dropped),
        "kept": len(rows) - len(dropped),
        "by_match_type": {t: by_type.get(t, 0) for t in MATCH_TYPES},
        "by_official_set_non_exclusive": by_set,
        "by_set_combination": dict(combos),
        "matched_via_side": dict(
            Counter(
                side_label(
                    any(m is not None and m["side"] in ("fr", "both") for m in d["sets"].values()),
                    any(m is not None and m["side"] in ("en", "both") for m in d["sets"].values()),
                )
                for d in dropped
            )
        ),
        "examples": {
            "seed": seed,
            "examples": [{k: d[k] for k in ("fr", "en", "match_type", "matched")} for d in sample],
        },
    }


def audit_e2(captured: Mapping[str, Any], output: prepare.PipelineOutput) -> dict[str, Any]:
    """Section (f): E2's removal step = 1000 held out + N removed as duplicate/near-dup of an E2
    item. Re-derived from the post-guard pool and the pipeline's own E2 sample."""
    e2_fr = output.protected_strings["e2"][: len(output.eval_rows["e2"])]
    e2_en = output.protected_strings["e2"][len(output.eval_rows["e2"]) :]
    e2_pairs = set(zip(e2_fr, e2_en, strict=True))
    matcher = SetMatcher(e2_fr + e2_en)
    e2_ids = {
        s: f"e2_{i:05d}" for i, s in reversed(list(enumerate(e2_fr)))
    }  # first occurrence wins
    e2_ids_en = {s: f"e2_{i:05d}" for i, s in reversed(list(enumerate(e2_en)))}

    held_out = 0
    extras: list[dict[str, Any]] = []
    for r in captured["kept"]:
        if (r["fr"], r["en"]) in e2_pairs:
            held_out += 1
            continue
        exact_hit, near_hit = pair_hits(r["fr"], r["en"], matcher.index)
        if exact_hit or near_hit:
            m = matcher.match(r["fr"], r["en"])
            assert m is not None
            targets: dict[str, list[str]] = {}
            for side, text in (("fr", r["fr"]), ("en", r["en"])):
                hits = matcher.targets_for(text)
                if hits:
                    targets[side] = hits
            # which E2 item(s) those target strings belong to
            e2_item_ids = sorted(
                {
                    (e2_ids if side == "fr" else e2_ids_en).get(t, "?")
                    for side, ts in targets.items()
                    for t in ts
                }
            )
            extras.append(
                {
                    "train_fr": r["fr"],
                    "train_en": r["en"],
                    "match_type": "exact" if exact_hit else "near_dup",
                    "side": m["side"],
                    "matched_e2_items": e2_item_ids,
                    "matched_strings": targets,
                    "e2_item_pairs": [
                        {
                            "id": i,
                            "fr": e2_fr[int(i.split("_")[1])],
                            "en": e2_en[int(i.split("_")[1])],
                        }
                        for i in e2_item_ids
                        if i != "?"
                    ],
                }
            )
    return {
        "post_guard_pool": len(captured["kept"]),
        "removed_total": held_out + len(extras),
        "held_out_e2_sample": held_out,
        "removed_as_duplicate_or_near_dup_of_an_e2_item": len(extras),
        "extra_items": extras,
    }


def reconcile(audit: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Assert every headline number equals data_manifest.json. Raises AssertionError listing each
    mismatch; returns the checked (name, audit value, manifest value) table when all agree."""
    t = audit["guard"]["totals"]
    lg = manifest["leakage_guard"]
    filters = {f["step"]: f for f in manifest["filters"]}
    guard_step = filters["leakage_guard_vs_dev_test_e1_e3"]
    e2_step = filters["e2_holdout_and_near_dup_removal"]
    checks = {
        "guard.removed": (t["removed"], guard_step["removed"]),
        "guard.pool_size_entering_guard": (
            t["pool_size_entering_guard"],
            guard_step["remaining"] + guard_step["removed"],
        ),
        "guard.exact": (t["exact"], lg["exact_hits"]),
        "guard.near_dup_only": (t["near_dup_only"], lg["near_dup_only_hits"]),
        "guard.exact+near==removed": (t["exact"] + t["near_dup_only"], guard_step["removed"]),
        "e1.dropped": (
            audit["e1"]["dropped"],
            manifest["eval_sets"]["e1"]["raw"] - manifest["eval_sets"]["e1"]["kept"],
        ),
        "e1.dropped_exact": (
            audit["e1"]["by_match_type"]["exact"],
            manifest["eval_sets"]["e1"]["dropped_leak_exact"],
        ),
        "e1.dropped_near_dup": (
            audit["e1"]["by_match_type"]["near_dup"],
            manifest["eval_sets"]["e1"]["dropped_leak_near_dup"],
        ),
        "e2.removed_total": (audit["e2"]["removed_total"], e2_step["removed"]),
        "e2.held_out": (audit["e2"]["held_out_e2_sample"], manifest["e2"]["removed_as_e2_holdout"]),
        "e2.extra": (
            audit["e2"]["removed_as_duplicate_or_near_dup_of_an_e2_item"],
            manifest["e2"]["removed_as_e2_near_dup_or_duplicate"],
        ),
    }
    for name in FINE_SETS:
        checks[f"guard.per_source_hits.{name}"] = (
            t["per_source_hits"][name],
            lg["per_source_hits"][name],
        )
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise AssertionError(f"audit does not reconcile with data_manifest.json: {bad}")
    return {k: {"audit": a, "manifest": m, "match": a == m} for k, (a, m) in checks.items()}


# --------------------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------------------


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _set_table(breakdown: Mapping[str, Any]) -> str:
    rows = []
    for name in COARSE_SETS:
        c = breakdown[name]
        rows.append(
            [
                name,
                c["total"],
                c["by_match_type"]["exact"],
                c["by_match_type"]["near_dup"],
                c["by_side"]["fr"],
                c["by_side"]["en"],
                c["by_side"]["both"],
            ]
        )
    return _md_table(
        ["set", "total", "exact", "near-dup only", "side fr", "side en", "side both"], rows
    )


def render_markdown(audit: Mapping[str, Any]) -> str:
    g, e1, e2 = audit["guard"], audit["e1"], audit["e2"]
    t = g["totals"]
    wb = g["by_french_word_count"]
    out = [
        "# Leakage audit (read-only)",
        "",
        "Re-derived by running `nmt.data.prepare.build_pipeline` on the cached datasets at the "
        "revisions pinned in `data/data_manifest.json`; every total below was asserted equal to "
        "that manifest (see `reconciliation` in `leakage_audit.json`). Generated by "
        "`scripts/audit_leakage.py`; no data artifact or rule was changed.",
        "",
        "## (a) Totals",
        "",
        _md_table(
            ["quantity", "value"],
            [
                [
                    "pool entering the guard (train pairs after steps 1-6)",
                    t["pool_size_entering_guard"],
                ],
                ["removed", t["removed"]],
                ["exact", t["exact"]],
                ["near-dup only", t["near_dup_only"]],
                *[[f"per-source hits: {k}", v] for k, v in t["per_source_hits"].items()],
            ],
        ),
        "",
        "## (b) By matched target set",
        "",
        "Non-exclusive (a pair counts in every set it matches; match type is per set; "
        "E2 is not part of the guard so it is 0):",
        "",
        _set_table(g["by_target_set_non_exclusive"]),
        "",
        f"Exclusive by first match, priority {' > '.join(PRIORITY)} (match type is pair-level):",
        "",
        _set_table(g["by_target_set_exclusive"]),
        "",
        "Side = which side(s) of the train pair matched.",
        "",
        "## (c) By train French word count (whitespace split, normalized)",
        "",
        _md_table(
            ["bucket", "exact", "near-dup only", "total", "pool size", "removal rate"],
            [
                [
                    b,
                    wb["by_bucket_and_match_type"][b]["exact"],
                    wb["by_bucket_and_match_type"][b]["near_dup"],
                    wb["by_bucket_and_match_type"][b]["total"],
                    wb["pool_bucket_counts"][b],
                    f"{100 * wb['removal_rate_per_bucket'][b]:.4f}%",
                ]
                for b in WORD_BUCKETS
            ],
        ),
        "",
        f"Share of near-dup-only removals that are 1-3-word sentences: "
        f"**{100 * wb['share_of_near_dup_removals_that_are_1_3_words']:.2f}%** "
        f"(1-3-word sentences are {100 * wb['pool_bucket_share']['1-3']:.2f}% of the pool).",
        "",
        f"## (d) 20 random near-dup removal examples (seed {g['near_dup_examples']['seed']})",
        "",
        _md_table(
            ["#", "train fr", "train en", "words", "matched set(s) / side", "matched target(s)"],
            [
                [
                    i,
                    ex["train_fr"].replace("|", "\\|"),
                    ex["train_en"].replace("|", "\\|"),
                    ex["word_count"],
                    "; ".join(f"{m['set']}/{m['side']}" for m in ex["matched"]),
                    "; ".join(
                        f"{m['set']}:{side}={tg}".replace("|", "\\|")
                        for m in ex["matched"]
                        for side, tg in m["targets"].items()
                    ),
                ]
                for i, ex in enumerate(g["near_dup_examples"]["examples"], 1)
            ],
        ),
        "",
        "## (e) E1: why 2000 opus-100 validation rows became 1940",
        "",
        f"{e1['dropped']} of {e1['raw_validation_rows']} validation rows match the protected set "
        f"P (dev source / dev reference / test source): exact={e1['by_match_type']['exact']}, "
        f"near-dup only={e1['by_match_type']['near_dup']}.",
        "",
        _md_table(
            ["official set", "total", "exact", "near-dup only"],
            [
                [k, v["total"], v["exact"], v["near_dup"]]
                for k, v in e1["by_official_set_non_exclusive"].items()
            ],
        ),
        "",
        "By set combination: "
        + ", ".join(f"{k}: {v}" for k, v in e1["by_set_combination"].items()),
        "",
        _md_table(
            ["#", "val fr", "val en", "type", "matched"],
            [
                [
                    i,
                    ex["fr"].replace("|", "\\|"),
                    ex["en"].replace("|", "\\|"),
                    ex["match_type"],
                    "; ".join(f"{m['set']}/{m['side']}" for m in ex["matched"]),
                ]
                for i, ex in enumerate(e1["examples"]["examples"], 1)
            ],
        ),
        "",
        "## (f) E2: why the e2 step removed 1001 pairs",
        "",
        f"{e2['removed_total']} = {e2['held_out_e2_sample']} held out (the E2 sample) + "
        f"{e2['removed_as_duplicate_or_near_dup_of_an_e2_item']} removed as duplicate/near-dup "
        "of an E2 item.",
        "",
    ]
    for ex in e2["extra_items"]:
        out += [
            f"- train fr: `{ex['train_fr']}`",
            f"  - train en: `{ex['train_en']}`",
            f"  - match: {ex['match_type']} on side {ex['side']}; E2 item(s): "
            f"{', '.join(ex['matched_e2_items'])}",
        ]
        for item in ex["e2_item_pairs"]:
            out += [f"  - {item['id']} fr: `{item['fr']}`", f"  - {item['id']} en: `{item['en']}`"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def run_audit(
    repo_root: Path, seed: int, loader: Callable[..., Any] = load_cached_datasets
) -> dict[str, Any]:
    manifest = json.loads((repo_root / "data" / "data_manifest.json").read_text(encoding="utf-8"))
    raw = loader(manifest)
    output, captured = run_pipeline_with_capture(raw, repo_root, manifest["seed"])
    audit: dict[str, Any] = {
        "source": {
            "manifest": "data/data_manifest.json",
            "datasets": manifest["datasets"],
            "pipeline_seed": manifest["seed"],
            "audit_seed": seed,
            "normalization": manifest["normalization"],
        },
        "guard": audit_guard(captured, output, seed),
        "e1": audit_e1(output, raw, seed),
        "e2": audit_e2(captured, output),
    }
    audit["reconciliation"] = reconcile(audit, manifest)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only leakage audit (spec section 3).")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "audit")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the random examples")
    args = parser.parse_args(argv)
    audit = run_audit(REPO_ROOT, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out_dir / "leakage_audit.md").write_text(
        render_markdown(audit), encoding="utf-8", newline="\n"
    )
    t = audit["guard"]["totals"]
    print(
        f"audit ok: removed={t['removed']} exact={t['exact']} near_dup_only={t['near_dup_only']} "
        f"per_source={t['per_source_hits']} (all reconciled with data_manifest.json)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
