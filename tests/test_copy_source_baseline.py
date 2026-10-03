from __future__ import annotations

# Offline tests for scripts/copy_source_baseline.py: copying scores BLEU 100 against a reference
# identical to the source (through the official-scorer wrapper), determinism, the committed
# artifacts carry the models' eval.json schema and are exactly the sources, and the calibration
# math on a toy example.
import json
from pathlib import Path

import pytest

from nmt.evaluate import REPO_ROOT, load_split, run_official_scorer, score_split_predictions
from scripts.copy_source_baseline import (
    RUN,
    copy_predictions,
    hyp_copy_level,
    reference_copy_level,
)
from scripts.eval_local import SPLITS

FINAL = REPO_ROOT / "reports" / "final"
needs_artifacts = pytest.mark.skipif(
    not (FINAL / RUN / "eval.json").is_file(), reason="baseline artifacts not generated yet"
)
TOY_INPUTS = [
    {"id": "t1", "source": "Le gouvernement de Paris a decide", "slice": "toy"},
    {"id": "t2", "source": "Il fait beau aujourd hui a Montreal", "slice": "toy"},
]


def _toy_labels(refs: list[str]) -> list[dict[str, str]]:
    return [
        {"id": r["id"], "reference": ref, "slice": "toy"}
        for r, ref in zip(TOY_INPUTS, refs, strict=True)
    ]


def test_copy_predictions_is_the_source_verbatim() -> None:
    pred = copy_predictions(TOY_INPUTS)
    assert pred == {r["id"]: r["source"] for r in TOY_INPUTS}


def test_copy_scores_bleu_100_against_identical_reference_via_official_wrapper(
    tmp_path: Path,
) -> None:
    inputs = [{"id": f"s{i}", "source": f"Phrase numero {i} avec des mots Paris"} for i in range(5)]
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps({"id": r["id"], "reference": r["source"], "slice": "toy"}) for r in inputs
        )
        + "\n",
        encoding="utf-8",
    )
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(copy_predictions(inputs)), encoding="utf-8")
    out = tmp_path / "out.json"
    run_official_scorer(gold, pred, out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all"]["bleu"] == 100.0
    assert report["all"]["chrf"] == 100.0


def test_scoring_is_deterministic() -> None:
    labels = _toy_labels(["The government of Paris decided", "It is nice today in Montreal"])
    pred = copy_predictions(TOY_INPUTS)
    a = score_split_predictions("e1", TOY_INPUTS, labels, pred, n_bootstrap=50, seed=1234)
    b = score_split_predictions("e1", TOY_INPUTS, labels, pred, n_bootstrap=50, seed=1234)
    assert a == b


@needs_artifacts
def test_committed_predictions_equal_the_sources() -> None:
    for split in SPLITS:
        inputs, _ = load_split(split)
        pred = json.loads((FINAL / RUN / f"{split}_predictions.json").read_text(encoding="utf-8"))
        assert pred == copy_predictions(inputs), split


@needs_artifacts
def test_eval_json_schema_equals_the_models() -> None:
    base = json.loads((FINAL / RUN / "eval.json").read_text(encoding="utf-8"))
    model = json.loads((FINAL / "main" / "seg_tuned" / "eval.json").read_text(encoding="utf-8"))
    assert list(base) == list(model)
    assert list(base["sets"]) == list(model["sets"])
    for split in model["sets"]:
        assert set(base["sets"][split]) == set(model["sets"][split]), split
    assert set(base["length_buckets_e1_e2_e3"]) == set(model["length_buckets_e1_e2_e3"])
    assert set(base["decoding_config"]) == set(model["decoding_config"])
    assert base["decoding_config"]["n_bootstrap"] == 1000
    assert base["decoding_config"]["bootstrap_seed"] == 1234


def test_reference_copy_level_toy() -> None:
    rows = [
        # the reference keeps "Paris" (equal to a source word); "of"/"The" are shorter than 4
        {"source": "Le gouvernement de Paris", "reference": "The government of Paris"},
        # nothing shared
        {"source": "Il fait beau", "reference": "It is nice today"},
    ]
    lvl = reference_copy_level(rows)
    # reference words of length >= 4: government, Paris | nice, today -> 4; shared: Paris -> 1
    assert lvl["ref_words_ge_min"] == 4
    assert lvl["ref_words_equal_to_source_word"] == 1
    assert lvl["ref_word_share"] == 0.25
    assert lvl["ref_sentences_with_any"] == 1
    assert lvl["ref_sentence_share"] == 0.5


def test_reference_copy_level_empty_group_is_none_not_zero() -> None:
    lvl = reference_copy_level([])
    assert lvl["ref_word_share"] is None and lvl["ref_sentence_share"] is None


def test_hyp_copy_level_toy_copy_vs_translation() -> None:
    rows = [
        {
            "source": "Le gouvernement de Paris",
            "reference": "The government of Paris",
            "hyp": "Le gouvernement de Paris",
        },
        {"source": "Il fait beau", "reference": "It is nice today", "hyp": "It is nice today"},
    ]
    lvl = hyp_copy_level(rows)
    # copy row: "gouvernement" is flagged (source word, not in the reference); "Paris" is in the
    # reference so it is not; "Le"/"de" are shorter than 4. Translation row: nothing flagged.
    assert lvl["word_copy_sentences"] == 1
    assert lvl["word_copy_sentence_rate"] == 0.5
    assert lvl["word_copy_words"] == 1
    # hypothesis words of length >= 4: gouvernement, Paris | nice, today
    assert lvl["hyp_words_ge_min"] == 4
    assert lvl["word_copy_word_share"] == 0.25
