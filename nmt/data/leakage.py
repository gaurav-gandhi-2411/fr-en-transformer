from __future__ import annotations

# Leakage-guard primitives shared by prepare.py's train-vs-eval-sets guard (spec §3,
# step 7) and by `check_overlap`, which independently re-reads `train.jsonl` from disk
# to verify zero overlap against dev/test/E1/E2/E3 (spec §12, PLAN.md interface note).
#
# A pair "hits" a protected source if its normalized fr OR en string either exactly
# equals one of the source's strings, or its near-duplicate key (see normalize.py)
# equals the near-duplicate key of one of the source's strings. Empty near-dup keys
# never match (an all-punctuation/all-digit string never near-dup-matches another).
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from nmt.data.normalize import near_dup_key


@dataclass(frozen=True)
class MatchIndex:
    """Exact strings and near-dup keys drawn from one protected source (e.g. dev
    sources, or the union of everything). Built once, queried many times."""

    exact: frozenset[str]
    keys: frozenset[str]

    @classmethod
    def from_strings(cls, strings: Iterable[str]) -> MatchIndex:
        """Build an index from normalized strings. Empty strings are ignored (they
        cannot leak anything and would otherwise poison the near-dup key set)."""
        exact: set[str] = set()
        keys: set[str] = set()
        for s in strings:
            if not s:
                continue
            exact.add(s)
            k = near_dup_key(s)
            if k:
                keys.add(k)
        return cls(exact=frozenset(exact), keys=frozenset(keys))

    def hit(self, s: str) -> tuple[bool, bool]:
        """Return (exact_hit, near_dup_hit) for normalized string `s` against this index."""
        if not s:
            return False, False
        exact_hit = s in self.exact
        key = near_dup_key(s)
        near_dup_hit = bool(key) and key in self.keys
        return exact_hit, near_dup_hit

    def __or__(self, other: MatchIndex) -> MatchIndex:
        return MatchIndex(exact=self.exact | other.exact, keys=self.keys | other.keys)


EMPTY_INDEX = MatchIndex(exact=frozenset(), keys=frozenset())


def pair_hits(fr: str, en: str, index: MatchIndex) -> tuple[bool, bool]:
    """Return (exact_hit, near_dup_hit) for a (fr, en) pair: a hit on either side counts."""
    fr_exact, fr_near = index.hit(fr)
    en_exact, en_near = index.hit(en)
    return fr_exact or en_exact, fr_near or en_near


def scan_and_filter(
    rows: list[dict],
    fr_key: str,
    en_key: str,
    indices: Mapping[str, MatchIndex],
) -> tuple[list[dict], dict]:
    """Remove every row whose normalized fr/en (looked up via `fr_key`/`en_key`) hits
    the union of `indices`. Returns (kept_rows, stats).

    `stats` = {"exact_hits", "near_dup_only_hits", "total_removed",
    "per_source_hits"} where `per_source_hits` counts, for each name in `indices`,
    how many *removed* rows hit that specific source -- a row hitting several sources
    increments each of their counters (spec §3 leakage-guard reporting).
    """
    union = EMPTY_INDEX
    for idx in indices.values():
        union = union | idx

    kept: list[dict] = []
    exact_hits = 0
    near_dup_only = 0
    per_source = dict.fromkeys(indices, 0)

    for row in rows:
        fr, en = row[fr_key], row[en_key]
        exact_hit, near_dup_hit = pair_hits(fr, en, union)
        if not (exact_hit or near_dup_hit):
            kept.append(row)
            continue
        if exact_hit:
            exact_hits += 1
        else:
            near_dup_only += 1
        for name, idx in indices.items():
            source_exact, source_near = pair_hits(fr, en, idx)
            if source_exact or source_near:
                per_source[name] += 1

    stats = {
        "exact_hits": exact_hits,
        "near_dup_only_hits": near_dup_only,
        "total_removed": exact_hits + near_dup_only,
        "per_source_hits": per_source,
    }
    return kept, stats


def check_overlap(train_path: str | Path, protected_strings: Mapping[str, Iterable[str]]) -> dict:
    """Independently re-read `train_path` (jsonl of `{"id","fr","en"}`, already
    normalized) and count overlap against each named protected source in
    `protected_strings` (values are already-normalized strings, e.g. dev sources,
    test sources, dev references, E1/E2/E3 sentences). Returns the same stats shape
    as `scan_and_filter`; a correct pipeline must show 0 for every count. This is the
    post-hoc, from-disk verification distinct from the in-memory guard in prepare.py.
    """
    indices = {
        name: MatchIndex.from_strings(strings) for name, strings in protected_strings.items()
    }
    rows: list[dict] = []
    with Path(train_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"fr": obj["fr"], "en": obj["en"]})
    _, stats = scan_and_filter(rows, "fr", "en", indices)
    return stats
