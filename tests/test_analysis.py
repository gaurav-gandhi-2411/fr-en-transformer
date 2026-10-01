from __future__ import annotations

# Tests for nmt/analysis.py: per-sentence features, gap decomposition, metric-artifact share,
# figure generation, example selection, and a small end-to-end run_analysis. Spec §10, §12.
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm

from nmt.analysis import (
    SentenceFeatures,
    dialogue_punct_density,
    gap_decomposition,
    is_truncated,
    length_ratio,
    metric_artifact_share,
    normalize_reference_for_artifact_check,
    proper_noun_copy_accuracy,
    proper_nouns,
    repetition_rate,
    run_analysis,
    select_worst_examples,
    source_rarity,
    untranslated_copy_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TOKENIZER = REPO_ROOT / "tokenizer" / "spm.model"
REAL_FREQ_SRC = REPO_ROOT / "tokenizer" / "freq_src.npy"
REAL_FREQ_TGT = REPO_ROOT / "tokenizer" / "freq_tgt.npy"

pytestmark = pytest.mark.skipif(
    not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)"
)


def test_length_ratio() -> None:
    assert length_ratio("a b c", "a b c") == 1.0
    assert length_ratio("a b", "a b c d") == pytest.approx(0.5)
    assert length_ratio("", "") == 1.0
    assert length_ratio("a", "") == 10.0  # empty-reference sentinel, not inf


def test_repetition_rate_detects_duplicate_trigrams() -> None:
    assert repetition_rate("a b c a b c") == pytest.approx(1 / 4)  # 1 dup among 4 trigrams
    assert repetition_rate("a b c d e") == 0.0
    assert repetition_rate("a b") == 0.0  # fewer than 3 words


def test_is_truncated() -> None:
    assert is_truncated("a b", "a b c d e f") is True
    assert is_truncated("a b c d e f", "a b c d e f") is False
    assert is_truncated("a", "") is False


def test_proper_nouns_skips_first_word() -> None:
    assert proper_nouns("Le grand Meaulnes est parti") == ["Meaulnes"]
    assert proper_nouns("Bonjour tout le monde") == []


def test_proper_noun_copy_accuracy() -> None:
    assert proper_noun_copy_accuracy("Le grand Meaulnes est parti", "Big Meaulnes left") == 1.0
    assert proper_noun_copy_accuracy("Le grand Meaulnes est parti", "Someone left") == 0.0
    assert proper_noun_copy_accuracy("bonjour tout le monde", "hello everyone") is None


def test_dialogue_punct_density() -> None:
    assert dialogue_punct_density("") == 0.0
    assert dialogue_punct_density("- Bonjour ! - Ca va ?") > 0.0


def test_normalize_reference_for_artifact_check_strips_trailing_quote() -> None:
    assert normalize_reference_for_artifact_check("I choose him.'") == "I choose him."
    assert normalize_reference_for_artifact_check('He said "hi"') == 'He said "hi'


def test_metric_artifact_share_reports_delta() -> None:
    rows = [{"id": "a", "hyp": "I choose him.", "reference": "I choose him.'"}]
    result = metric_artifact_share(rows)
    assert result["n"] == 1
    assert result["chrf_ref_normalized"] >= result["chrf_original"]
    assert result["delta_chrf"] == pytest.approx(
        result["chrf_ref_normalized"] - result["chrf_original"]
    )


def test_source_rarity_and_untranslated_copy_rate_use_real_tokenizer() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(REAL_TOKENIZER))
    freq_src = np.load(REAL_FREQ_SRC)
    freq_tgt = np.load(REAL_FREQ_TGT)
    byte_fallback_ids: set[int] = set()

    rarity = source_rarity("Bonjour le monde.", sp, freq_src, byte_fallback_ids)
    assert rarity["mean_freq"] >= 0.0
    assert rarity["min_freq"] >= 0.0

    # A hypothesis containing a token that appears verbatim in the source and was never seen on
    # the train target side (freq_tgt == 0 for every piece id, since freq_tgt is all zeros here)
    # must be flagged as fully copied.
    zero_freq_tgt = np.zeros_like(freq_tgt)
    rate = untranslated_copy_rate("Bonjour", "Bonjour le monde.", sp, zero_freq_tgt)
    assert rate == 1.0


def _synthetic_features(n_e1: int, n_e3: int) -> list[SentenceFeatures]:
    feats = []
    rng = np.random.default_rng(0)
    for i in range(n_e1):
        feats.append(
            SentenceFeatures(
                id=f"e1_{i}",
                domain="e1",
                length_ratio=float(rng.normal(1.0, 0.05)),
                repetition_rate=0.0,
                truncated=False,
                untranslated_copy_rate=0.0,
                src_rarity_mean=50.0,
                src_rarity_min=10.0,
                src_byte_fallback_rate=0.0,
                proper_noun_copy_accuracy=1.0,
                dialogue_punct_density=0.05,
                chrf=float(80 + rng.normal(0, 2)),
            )
        )
    for i in range(n_e3):
        feats.append(
            SentenceFeatures(
                id=f"e3_{i}",
                domain="e3",
                length_ratio=float(rng.normal(0.8, 0.1)),
                repetition_rate=0.1,
                truncated=False,
                untranslated_copy_rate=0.05,
                src_rarity_mean=5.0,
                src_rarity_min=1.0,
                src_byte_fallback_rate=0.02,
                proper_noun_copy_accuracy=0.5,
                dialogue_punct_density=0.15,
                chrf=float(60 + rng.normal(0, 2)),
            )
        )
    return feats


def test_gap_decomposition_total_gap_matches_group_means() -> None:
    feats = _synthetic_features(30, 30)
    result = gap_decomposition(feats)
    e1_mean = np.mean([f.chrf for f in feats if f.domain == "e1"])
    e3_mean = np.mean([f.chrf for f in feats if f.domain == "e3"])
    assert result["total_gap_chrf_e1_minus_e3"] == pytest.approx(e1_mean - e3_mean, abs=1e-6)
    assert result["n"] == 60
    assert set(result["share_of_gap"]) == {
        "length_dev",
        "repetition_rate",
        "rarity",
        "dialogue_punct_density",
        "residual_domain",
    }
    # contributions should sum (approximately) back to the total gap, by construction of a
    # single-regression linear decomposition.
    assert sum(result["contributions_chrf"].values()) == pytest.approx(
        result["total_gap_chrf_e1_minus_e3"], abs=1e-6
    )


def test_select_worst_examples_picks_lowest_chrf_per_slice() -> None:
    feats = [
        SentenceFeatures("d1", "dev", 1.0, 0.0, False, 0.0, 1, 1, 0.0, 1.0, 0.0, chrf=90.0),
        SentenceFeatures("d2", "dev", 1.0, 0.0, False, 0.0, 1, 1, 0.0, 1.0, 0.0, chrf=10.0),
        SentenceFeatures("d3", "dev", 1.0, 0.0, False, 0.0, 1, 1, 0.0, 1.0, 0.0, chrf=50.0),
    ]
    rows_by_id = {
        "d1": {"slice": "seen", "source": "s1", "reference": "r1", "hyp": "h1"},
        "d2": {"slice": "seen", "source": "s2", "reference": "r2", "hyp": "h2"},
        "d3": {"slice": "seen", "source": "s3", "reference": "r3", "hyp": "h3"},
    }
    examples = select_worst_examples(feats, rows_by_id, n_per_slice=2)
    assert [e["id"] for e in examples] == ["d2", "d3"]  # lowest chrF first


_TINY_SPLITS = {
    "e1": (
        [
            {
                "id": f"e1_{i}",
                "source": f"Phrase source numero {i} avec des mots.",
                "slice": "e1",
                "length": 30,
            }
            for i in range(6)
        ],
        [
            {"id": f"e1_{i}", "reference": f"Source sentence number {i} with words.", "slice": "e1"}
            for i in range(6)
        ],
    ),
    "e3": (
        [
            {
                "id": f"e3_{i}",
                "source": f"Une autre phrase litteraire numero {i}.",
                "slice": "e3",
                "length": 30,
            }
            for i in range(6)
        ],
        [
            {"id": f"e3_{i}", "reference": f"Another literary sentence number {i}.", "slice": "e3"}
            for i in range(6)
        ],
    ),
    "dev": (
        [
            {"id": "dev_0", "source": "Bonjour.", "slice": "seen", "length": 8},
            {"id": "dev_1", "source": "Au revoir.", "slice": "seen", "length": 10},
            {"id": "dev_2", "source": "Il pleut.", "slice": "unseen_domain", "length": 9},
            {"id": "dev_3", "source": "Merci beaucoup.", "slice": "unseen_domain", "length": 15},
        ],
        [
            {"id": "dev_0", "reference": "Hello.", "slice": "seen"},
            {"id": "dev_1", "reference": "Goodbye.", "slice": "seen"},
            {"id": "dev_2", "reference": "It's raining.", "slice": "unseen_domain"},
            {"id": "dev_3", "reference": "Thank you very much.", "slice": "unseen_domain"},
        ],
    ),
}


def test_run_analysis_end_to_end_tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmt.analysis.load_split", lambda name: _TINY_SPLITS[name])
    e1_pred = {r["id"]: r["reference"] for r in _TINY_SPLITS["e1"][1]}
    e3_pred = {r["id"]: "completely wrong output" for r in _TINY_SPLITS["e3"][1]}
    dev_pred = {r["id"]: r["reference"] for r in _TINY_SPLITS["dev"][1]}

    out_dir = tmp_path / "analysis_out"
    result = run_analysis(e1_pred, e3_pred, dev_pred, out_dir)

    assert (out_dir / "analysis.json").is_file()
    assert (out_dir / "examples.json").is_file()
    for fig_name in (
        "chrf_vs_length_bucket.png",
        "chrf_vs_rarity_decile.png",
        "failure_mode_rates.png",
    ):
        assert (out_dir / "figures" / fig_name).is_file()
    assert result["n_e1"] == 6
    assert result["n_e3"] == 6
    assert result["n_dev"] == 4
    assert "gap_decomposition" in result
    assert "metric_artifact_share" in result

    import json

    examples = json.loads((out_dir / "examples.json").read_text(encoding="utf-8"))
    assert len(examples) <= 4  # 2 dev slices present x up to 2 each
    assert {e["slice"] for e in examples} <= {"seen", "unseen_domain"}
