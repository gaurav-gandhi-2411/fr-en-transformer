from __future__ import annotations

# Leakage-guard unit tests on synthetic data (spec §3 step 7, §12), plus a real-data
# check of `check_overlap` against the committed eval sets / dev / test files and the
# real `data/processed/train.jsonl`, skipped (with an explicit reason -- the only
# allowed skip) if that file hasn't been produced yet in this checkout.
import json
from pathlib import Path

import pytest

from nmt.data.leakage import MatchIndex, check_overlap, pair_hits, scan_and_filter
from nmt.data.normalize import near_dup_key, normalize_text

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.jsonl"


def test_match_index_exact_hit() -> None:
    idx = MatchIndex.from_strings(["Bonjour le monde", "Au revoir"])
    exact, near_dup = idx.hit("Bonjour le monde")
    assert exact is True
    assert near_dup is True  # an exact hit also has matching near-dup key


def test_match_index_no_hit() -> None:
    idx = MatchIndex.from_strings(["Bonjour le monde"])
    exact, near_dup = idx.hit("Something else entirely")
    assert exact is False
    assert near_dup is False


def test_match_index_empty_string_never_hits() -> None:
    idx = MatchIndex.from_strings(["Bonjour le monde"])
    exact, near_dup = idx.hit("")
    assert exact is False
    assert near_dup is False


def test_match_index_empty_key_never_matches() -> None:
    """A protected string with no alphanumeric characters contributes no near-dup key,
    so it can never near-dup-match anything -- not even another punctuation-only string."""
    idx = MatchIndex.from_strings(["!!! ??? ..."])
    assert near_dup_key("!!! ??? ...") == ""
    exact, near_dup = idx.hit("... !!! ???")
    assert exact is False
    assert near_dup is False


def test_pair_hits_exact_on_fr_side() -> None:
    idx = MatchIndex.from_strings(["Bonjour le monde"])
    exact, near_dup = pair_hits("Bonjour le monde", "Hello world", idx)
    assert exact is True


def test_pair_hits_exact_on_en_side() -> None:
    idx = MatchIndex.from_strings(["Hello world"])
    exact, near_dup = pair_hits("Bonjour le monde", "Hello world", idx)
    assert exact is True


def test_pair_hits_near_dup_via_case_and_punctuation_differences() -> None:
    idx = MatchIndex.from_strings(["Bonjour, le monde !"])
    exact, near_dup = pair_hits("BONJOUR LE MONDE", "Hello world", idx)
    assert exact is False
    assert near_dup is True


def test_scan_and_filter_removes_hits_and_counts_per_source() -> None:
    rows = [
        {"fr": "Bonjour le monde", "en": "Hello world"},  # exact hit on dev_src
        {"fr": "Bonsoir", "en": "Good evening"},  # near-dup hit on test_src ("BONSOIR!!")
        {"fr": "Complètement différent", "en": "Totally different"},  # no hit
    ]
    indices = {
        "dev_src": MatchIndex.from_strings(["Bonjour le monde"]),
        "test_src": MatchIndex.from_strings(["BONSOIR!!"]),
    }
    kept, stats = scan_and_filter(rows, "fr", "en", indices)
    assert [r["fr"] for r in kept] == ["Complètement différent"]
    assert stats["exact_hits"] == 1
    assert stats["near_dup_only_hits"] == 1
    assert stats["total_removed"] == 2
    assert stats["per_source_hits"] == {"dev_src": 1, "test_src": 1}


def test_scan_and_filter_row_can_hit_multiple_sources() -> None:
    rows = [{"fr": "Bonjour le monde", "en": "Hello world"}]
    indices = {
        "dev_src": MatchIndex.from_strings(["Bonjour le monde"]),
        "e1": MatchIndex.from_strings(["Hello world"]),
    }
    kept, stats = scan_and_filter(rows, "fr", "en", indices)
    assert kept == []
    assert stats["per_source_hits"] == {"dev_src": 1, "e1": 1}


def test_scan_and_filter_after_guard_overlap_is_zero() -> None:
    """After scan_and_filter, re-scanning the kept rows against the same indices finds
    nothing -- i.e. the guard actually achieves zero overlap, not just partial removal."""
    rows = [
        {"fr": "Bonjour le monde", "en": "Hello world"},
        {"fr": "BONJOUR, LE MONDE", "en": "Something else"},  # near-dup of fr above
        {"fr": "Une phrase neuve", "en": "A brand new sentence"},
    ]
    indices = {"dev_src": MatchIndex.from_strings(["Bonjour le monde"])}
    kept, _ = scan_and_filter(rows, "fr", "en", indices)
    _, post_stats = scan_and_filter(kept, "fr", "en", indices)
    assert post_stats["total_removed"] == 0
    assert kept == [{"fr": "Une phrase neuve", "en": "A brand new sentence"}]


def test_check_overlap_on_synthetic_train_file(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    rows = [
        {"id": "train_0000000", "fr": "Bonjour le monde", "en": "Clean sentence one"},
        {"id": "train_0000001", "fr": "Une phrase propre", "en": "A clean sentence"},
        {"id": "train_0000002", "fr": "BONJOUR, LE MONDE !", "en": "Another clean one"},
    ]
    with train_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    protected = {"dev_src": ["Bonjour le monde"]}
    stats = check_overlap(train_path, protected)
    # row 0 is an exact hit; row 2 is a near-dup-only hit (case/punctuation differ).
    assert stats["exact_hits"] == 1
    assert stats["near_dup_only_hits"] == 1
    assert stats["total_removed"] == 2
    assert stats["per_source_hits"] == {"dev_src": 2}


def test_check_overlap_on_real_committed_data_is_zero() -> None:
    """Independently verify zero overlap between the real, currently-committed eval
    sets / dev / test files and the real (gitignored, locally-produced) train.jsonl.
    Skipped -- with this explicit reason, the only skip allowed in this suite -- when
    train.jsonl has not been produced yet in this checkout (e.g. a fresh clone before
    `python -m nmt.data.prepare` has been run)."""
    if not TRAIN_PATH.is_file():
        pytest.skip(f"{TRAIN_PATH} does not exist yet -- run `python -m nmt.data.prepare` first")

    def _field(path: Path, field: str) -> list[str]:
        values = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    values.append(json.loads(line)[field])
        return values

    dev_src = [
        normalize_text(s) for s in _field(REPO_ROOT / "data" / "dev" / "inputs.jsonl", "source")
    ]
    test_src = [
        normalize_text(s) for s in _field(REPO_ROOT / "data" / "test" / "inputs.jsonl", "source")
    ]
    dev_ref = [
        normalize_text(s) for s in _field(REPO_ROOT / "data" / "dev" / "labels.jsonl", "reference")
    ]

    protected: dict[str, list[str]] = {"dev_src": dev_src, "test_src": test_src, "dev_ref": dev_ref}
    for slice_name in ("e1", "e2", "e3"):
        inputs_path = REPO_ROOT / "data" / "eval" / slice_name / "inputs.jsonl"
        labels_path = REPO_ROOT / "data" / "eval" / slice_name / "labels.jsonl"
        sources = [normalize_text(s) for s in _field(inputs_path, "source")]
        references = [normalize_text(s) for s in _field(labels_path, "reference")]
        protected[slice_name] = sources + references

    stats = check_overlap(TRAIN_PATH, protected)
    assert stats["exact_hits"] == 0
    assert stats["near_dup_only_hits"] == 0
    assert stats["total_removed"] == 0
