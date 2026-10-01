from __future__ import annotations

# Data-pipeline unit and integration tests (spec §3, §12), entirely network-free:
# `build_pipeline` takes plain in-memory iterables, never touching `datasets` or the
# network (that lives only in `load_raw_datasets`/`prepare`, exercised by hand at the
# end of this phase, not in this fast test suite).
from nmt.data.prepare import (
    PipelineOutput,
    _drop_empty_side,
    _drop_exact_duplicates,
    _drop_fr_eq_en,
    _drop_high_non_letter,
    _drop_length_ratio,
    _length_ratio_ok,
    _non_letter_fraction,
    build_pipeline,
)

# --------------------------------------------------------------------------------------
# Per-filter unit tests on tiny synthetic lists (spec §3 order-of-operations steps 2-6)
# --------------------------------------------------------------------------------------


def test_drop_empty_side_removes_rows_with_either_side_empty() -> None:
    rows = [
        {"fr": "Bonjour", "en": "Hello"},
        {"fr": "", "en": "Hello"},
        {"fr": "Bonjour", "en": ""},
        {"fr": "Salut", "en": "Hi"},
    ]
    kept, removed = _drop_empty_side(rows)
    assert removed == 2
    assert [r["fr"] for r in kept] == ["Bonjour", "Salut"]


def test_drop_exact_duplicates_keeps_first_occurrence() -> None:
    rows = [
        {"fr": "Bonjour", "en": "Hello", "tag": "first"},
        {"fr": "Salut", "en": "Hi"},
        {"fr": "Bonjour", "en": "Hello", "tag": "second"},  # exact dup of row 0
    ]
    kept, removed = _drop_exact_duplicates(rows)
    assert removed == 1
    assert len(kept) == 2
    assert kept[0]["tag"] == "first"  # first occurrence kept, not the later duplicate


def test_drop_exact_duplicates_same_fr_different_en_is_not_a_duplicate() -> None:
    rows = [{"fr": "Bonjour", "en": "Hello"}, {"fr": "Bonjour", "en": "Good day"}]
    kept, removed = _drop_exact_duplicates(rows)
    assert removed == 0
    assert len(kept) == 2


def test_drop_fr_eq_en_removes_identical_sides() -> None:
    rows = [{"fr": "Paris", "en": "Paris"}, {"fr": "Bonjour", "en": "Hello"}]
    kept, removed = _drop_fr_eq_en(rows)
    assert removed == 1
    assert kept == [{"fr": "Bonjour", "en": "Hello"}]


def test_length_ratio_ok_boundaries() -> None:
    assert _length_ratio_ok("abc", "abc") is True  # ratio 1.0
    assert _length_ratio_ok("a", "aaa") is True  # ratio exactly 1/3
    assert _length_ratio_ok("aaa", "a") is True  # ratio exactly 3.0
    assert _length_ratio_ok("a", "aaaa") is False  # ratio 0.25 < 1/3
    assert _length_ratio_ok("aaaa", "a") is False  # ratio 4.0 > 3
    assert _length_ratio_ok("", "abc") is False  # empty side is defensively rejected


def test_drop_length_ratio_removes_out_of_bounds_pairs() -> None:
    rows = [
        {"fr": "Ah", "en": "This is unquestionably too long a sentence for ratio"},
        {"fr": "Bonjour tout va bien", "en": "Hello all is well"},
    ]
    kept, removed = _drop_length_ratio(rows)
    assert removed == 1
    assert kept == [{"fr": "Bonjour tout va bien", "en": "Hello all is well"}]


def test_non_letter_fraction_counts_over_non_whitespace_chars() -> None:
    assert _non_letter_fraction("Bonjour") == 0.0
    assert _non_letter_fraction("123456") == 1.0
    assert _non_letter_fraction("Bonjour!") == 1 / 8  # one '!' out of 8 non-space chars
    assert _non_letter_fraction("   ") == 1.0  # all-whitespace: defensively 100%


def test_drop_high_non_letter_removes_rows_over_50_percent() -> None:
    rows = [
        {"fr": "123456", "en": "Numbers here"},  # fr is 100% non-letter -> dropped
        {"fr": "Bonjour", "en": "9!!!!!!"},  # en is >50% non-letter -> dropped
        {"fr": "Bonjour", "en": "Hello"},  # both sides fine -> kept
    ]
    kept, removed = _drop_high_non_letter(rows)
    assert removed == 2
    assert kept == [{"fr": "Bonjour", "en": "Hello"}]


# --------------------------------------------------------------------------------------
# Full build_pipeline integration test: one worked example exercising every step
# (normalize, all 6 train filters, E1/E3 vs P, the leakage guard vs P+E1+E3 with a
# per-source hit on each of the 5 sources, and E2 holdout + near-dup removal).
# --------------------------------------------------------------------------------------

DEV_SOURCES = ["Bonjour tout le monde"]
TEST_SOURCES = ["Au revoir tout le monde"]
DEV_REFERENCES = ["Hello everyone"]

_LONG_FR = ("mot " * 60).strip()  # 239 chars: > 200, the sole E2 candidate
_LONG_EN = ("word " * 60).strip()  # 299 chars: ratio 239/299 is within [1/3, 3]

VAL_PAIRS = [
    {"fr": "Au revoir tout le monde", "en": "Goodbye everyone people"},  # leaks vs test_src
    {"fr": "Comment allez-vous", "en": "How are you doing"},  # clean -> becomes E1's one row
]
BOOKS_PAIRS = [
    {"fr": "Un livre totalement different", "en": "Hello everyone"},  # leaks vs dev_ref (en side)
    {"fr": "Le vieux marin racontait une histoire", "en": "The old sailor told a story"},  # -> E3
]
TRAIN_PAIRS = [
    {"fr": "Bonjour tout le monde", "en": "Hi there everyone people"},  # leaks vs dev_src
    {"fr": "   ", "en": "Something"},  # empty fr after normalize -> dropped
    {"fr": "Le chat noir", "en": "The black cat"},  # survives to final train
    {"fr": "Le chat noir", "en": "The black cat"},  # exact duplicate of the previous row
    {"fr": "Paris", "en": "Paris"},  # fr == en -> dropped
    {"fr": "Ah", "en": "This is unquestionably too long a sentence for ratio here"},  # bad ratio
    {"fr": "123456", "en": "Numbers here"},  # >50% non-letter on fr
    {"fr": "Il fait beau aujourd'hui", "en": "It is nice today"},  # survives to final train
    {"fr": _LONG_FR, "en": _LONG_EN},  # the sole E2 candidate; gets held out entirely
    {"fr": ("chat " * 30).strip(), "en": _LONG_EN},  # short fr, en duplicates the E2 sentence's en
    {"fr": "Comment allez-vous", "en": "Something unrelated entirely"},  # leaks vs E1
    {"fr": "Le vieux marin racontait une histoire", "en": "A completely different translation"},
]


def _run() -> PipelineOutput:
    return build_pipeline(
        train_pairs=TRAIN_PAIRS,
        val_pairs=VAL_PAIRS,
        books_pairs=BOOKS_PAIRS,
        dev_sources=DEV_SOURCES,
        test_sources=TEST_SOURCES,
        dev_references=DEV_REFERENCES,
        seed=1234,
    )


def test_build_pipeline_filters_table() -> None:
    output = _run()
    steps = [(f["step"], f["removed"], f["remaining"]) for f in output.stats["filters"]]
    assert steps == [
        ("normalize", 0, 12),
        ("drop_empty_side", 1, 11),
        ("drop_exact_duplicate_pairs", 1, 10),
        ("drop_fr_eq_en", 1, 9),
        ("drop_length_ratio_out_of_bounds", 1, 8),
        ("drop_high_non_letter_fraction", 1, 7),
        ("leakage_guard_vs_dev_test_e1_e3", 3, 4),
        ("e2_holdout_and_near_dup_removal", 2, 2),
    ]


def test_build_pipeline_leakage_guard_per_source_breakdown() -> None:
    output = _run()
    guard = output.stats["leakage_guard"]
    assert guard["exact_hits"] == 3
    assert guard["near_dup_only_hits"] == 0
    assert guard["per_source_hits"] == {"dev_src": 1, "test_src": 0, "dev_ref": 0, "e1": 1, "e3": 1}


def test_build_pipeline_e1_e3_stats() -> None:
    output = _run()
    e1 = output.stats["eval_sets"]["e1"]
    assert e1 == {
        "raw": 2,
        "dropped_empty": 0,
        "dropped_leak_exact": 1,
        "dropped_leak_near_dup": 0,
        "kept": 1,
        "sampled": 1,
    }
    e3 = output.stats["eval_sets"]["e3"]
    assert e3 == {
        "raw": 2,
        "dropped_empty": 0,
        "dropped_leak_exact": 1,
        "dropped_leak_near_dup": 0,
        "kept": 1,
        "sampled": 1,
    }


def test_build_pipeline_e2_holdout_vs_near_dup_counted_separately() -> None:
    output = _run()
    e2 = output.stats["e2"]
    assert e2["candidates_len_fr_gt_200"] == 1
    assert e2["sampled"] == 1
    assert e2["removed_as_e2_holdout"] == 1
    assert e2["removed_as_e2_near_dup_or_duplicate"] == 1


def test_build_pipeline_final_train_rows() -> None:
    output = _run()
    assert output.stats["final_train_pair_count"] == 2
    assert output.train_rows == [
        {"id": "train_0000000", "fr": "Le chat noir", "en": "The black cat"},
        {"id": "train_0000001", "fr": "Il fait beau aujourd'hui", "en": "It is nice today"},
    ]


def test_build_pipeline_eval_rows_format_and_content() -> None:
    output = _run()
    (e1_row,) = output.eval_rows["e1"]
    assert e1_row == {
        "id": "e1_00000",
        "source": "Comment allez-vous",
        "reference": "How are you doing",
        "slice": "e1",
        "length": len("Comment allez-vous"),
    }
    (e2_row,) = output.eval_rows["e2"]
    assert e2_row["id"] == "e2_00000"
    assert e2_row["source"] == _LONG_FR
    assert e2_row["reference"] == _LONG_EN
    assert e2_row["length"] == len(_LONG_FR)
    (e3_row,) = output.eval_rows["e3"]
    assert e3_row == {
        "id": "e3_00000",
        "source": "Le vieux marin racontait une histoire",
        "reference": "The old sailor told a story",
        "slice": "e3",
        "length": len("Le vieux marin racontait une histoire"),
    }


def test_build_pipeline_protected_strings_are_normalized_and_complete() -> None:
    output = _run()
    ps = output.protected_strings
    assert ps["dev_src"] == ["Bonjour tout le monde"]
    assert ps["test_src"] == ["Au revoir tout le monde"]
    assert ps["dev_ref"] == ["Hello everyone"]
    assert ps["e1"] == ["Comment allez-vous", "How are you doing"]
    assert ps["e3"] == ["Le vieux marin racontait une histoire", "The old sailor told a story"]
    assert ps["e2"] == [_LONG_FR, _LONG_EN]


def test_build_pipeline_is_deterministic_across_runs() -> None:
    """Same inputs + same seed => byte-identical output, every field (spec §3/§12:
    the real prepare run is checked for this at the file-hash level; this is the same
    property at the in-memory level)."""
    first = _run()
    second = _run()
    assert first == second
