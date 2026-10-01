from __future__ import annotations

# Tests for nmt/evaluate.py: the harness wrapper's numbers equal the vendored CLI's output
# exactly (spec §12), sacreBLEU signatures, bootstrap CI/paired-bootstrap statistical behavior
# on synthetic systems, the length-bucket view, and a small end-to-end run_evaluation using a
# tiny model. Spec §8, §12.
import json
from pathlib import Path

import pytest
import torch

from nmt.evaluate import (
    EvalRunConfig,
    bootstrap_ci_official,
    compute_official_metrics,
    length_bucket_label,
    length_bucket_view,
    paired_bootstrap,
    run_evaluation,
    run_official_scorer_cli,
    sacrebleu_metrics,
)
from nmt.hub import export_checkpoint
from nmt.model.transformer import ModelConfig, Transformer
from nmt.translate import Translator

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TOKENIZER = REPO_ROOT / "tokenizer" / "spm.model"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


_GOLD_ROWS = [
    {"id": "d1", "reference": "the cat sat on the mat", "slice": "seen"},
    {"id": "d2", "reference": "hello world, how are you?", "slice": "seen"},
    {
        "id": "d3",
        "reference": "a much longer reference sentence to vary bleu behavior a bit",
        "slice": "long",
    },
    {"id": "d4", "reference": "un chat noir", "slice": "unseen_domain"},
    {"id": "d5", "reference": "it was a dark and stormy night", "slice": "unseen_domain"},
]
_PRED = {
    "d1": "the cat sat on the mat",
    "d2": "hello world how are you",
    "d3": "a much shorter sentence",
    "d4": "a black cat",
    "d5": "it was dark and stormy",
}


def test_compute_official_metrics_matches_cli_exactly(tmp_path: Path) -> None:
    gold_path = tmp_path / "labels.jsonl"
    _write_jsonl(gold_path, _GOLD_ROWS)
    pred_path = tmp_path / "preds.json"
    pred_path.write_text(json.dumps(_PRED), encoding="utf-8")
    out_path = tmp_path / "cli_report.json"

    cli_report = run_official_scorer_cli(gold_path, pred_path, out_path)
    harness_report = compute_official_metrics(_PRED, _GOLD_ROWS)

    assert harness_report == cli_report


def test_sacrebleu_metrics_has_score_and_signature() -> None:
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]
    refs = [r["reference"] for r in _GOLD_ROWS]
    result = sacrebleu_metrics(hyps, refs)
    for key in ("bleu", "chrf", "chrf++"):
        assert isinstance(result[key]["score"], float)
        assert isinstance(result[key]["signature"], str) and result[key]["signature"]


def test_bootstrap_ci_brackets_point_estimate() -> None:
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]
    refs = [r["reference"] for r in _GOLD_ROWS]
    ci = bootstrap_ci_official(hyps, refs, "chrf", n_resamples=200, seed=1234)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n"] == len(_GOLD_ROWS)
    assert ci["n_resamples"] == 200


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]
    refs = [r["reference"] for r in _GOLD_ROWS]
    ci1 = bootstrap_ci_official(hyps, refs, "bleu", n_resamples=100, seed=7)
    ci2 = bootstrap_ci_official(hyps, refs, "bleu", n_resamples=100, seed=7)
    assert ci1 == ci2


def test_paired_bootstrap_detects_a_clearly_better_than_b() -> None:
    """System A = perfect hypotheses (== references), system B = obviously wrong. Delta must be
    positive and the one-sided p-value small (spec §12's "test on synthetic systems")."""
    refs = [r["reference"] for r in _GOLD_ROWS]
    hyps_a = list(refs)  # perfect
    hyps_b = ["completely unrelated garbage text zzz" for _ in refs]

    def chrf_fn(h: list[str], r: list[str]) -> float:
        from nmt.evaluate import load_official_module

        module = load_official_module()
        return (
            sum(module.chrf_sentence(a, b) for a, b in zip(h, r, strict=True)) / len(r)
            if r
            else 0.0
        )

    result = paired_bootstrap(hyps_a, hyps_b, refs, chrf_fn, n_resamples=500, seed=1234)
    assert result["delta"] > 0
    assert result["ci_low"] > 0  # entire CI on the positive side: A is robustly better
    assert result["p_value"] < 0.05


def test_paired_bootstrap_identical_systems_has_zero_delta() -> None:
    refs = [r["reference"] for r in _GOLD_ROWS]
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]

    def chrf_fn(h: list[str], r: list[str]) -> float:
        from nmt.evaluate import load_official_module

        module = load_official_module()
        return (
            sum(module.chrf_sentence(a, b) for a, b in zip(h, r, strict=True)) / len(r)
            if r
            else 0.0
        )

    result = paired_bootstrap(hyps, hyps, refs, chrf_fn, n_resamples=100, seed=1234)
    assert result["delta"] == 0.0


def test_length_bucket_label_boundaries() -> None:
    assert length_bucket_label(1) == "<=10"
    assert length_bucket_label(10) == "<=10"
    assert length_bucket_label(11) == "11-20"
    assert length_bucket_label(20) == "11-20"
    assert length_bucket_label(21) == "21-40"
    assert length_bucket_label(40) == "21-40"
    assert length_bucket_label(41) == "41-80"
    assert length_bucket_label(80) == "41-80"
    assert length_bucket_label(81) == ">80"
    assert length_bucket_label(500) == ">80"


def test_length_bucket_view_groups_by_word_count() -> None:
    rows = [
        {"id": "a", "source": "one two three", "reference": "one two three"},
        {"id": "b", "source": " ".join(["word"] * 15), "reference": "word " * 15},
        {"id": "c", "source": "another short one", "reference": "another short one"},
    ]
    pred = {"a": "one two three", "b": "word " * 15, "c": "another short one"}
    view = length_bucket_view(rows, pred)
    assert set(view.keys()) == {"<=10", "11-20"}
    assert view["<=10"]["n"] == 2
    assert view["11-20"]["n"] == 1


# ---- small end-to-end run_evaluation, monkeypatched to tiny in-memory eval splits ----

pytestmark_tokenizer = pytest.mark.skipif(
    not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)"
)


def _export_tiny_translator(tmp_path: Path) -> Translator:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16000, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32)
    model = Transformer(cfg)
    ckpt_path = tmp_path / "step_00000010.pt"
    torch.save({"step": 10, "model": model.state_dict()}, ckpt_path)
    out_dir = export_checkpoint([ckpt_path], tmp_path / "export", cfg, REAL_TOKENIZER, average=True)
    return Translator.from_pretrained(str(out_dir), device="cpu")


@pytestmark_tokenizer
def test_run_evaluation_end_to_end_tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_dev = (
        [
            {"id": "dev_0", "source": "Bonjour le monde.", "slice": "seen", "length": 20},
            {"id": "dev_1", "source": "Ceci est un test.", "slice": "unseen_domain", "length": 20},
        ],
        [
            {"id": "dev_0", "reference": "Hello world.", "slice": "seen"},
            {"id": "dev_1", "reference": "This is a test.", "slice": "unseen_domain"},
        ],
    )
    tiny_e1 = (
        [{"id": "e1_0", "source": "Au revoir.", "slice": "e1", "length": 10}],
        [{"id": "e1_0", "reference": "Goodbye.", "slice": "e1"}],
    )

    def fake_load_split(name: str) -> tuple[list[dict], list[dict]]:
        return {"dev": tiny_dev, "e1": tiny_e1}[name]

    monkeypatch.setattr("nmt.evaluate.load_split", fake_load_split)

    translator = _export_tiny_translator(tmp_path)
    decode_cfg = EvalRunConfig(
        beam_size=2, alpha=0.6, batch_size=4, n_bootstrap=20, bootstrap_seed=1
    )
    out_dir = tmp_path / "reports" / "smoke" / "ckpt0"
    result = run_evaluation(
        translator, "smoke", "ckpt0", decode_cfg, out_dir=out_dir, splits=("dev", "e1")
    )

    assert (out_dir / "eval.json").is_file()
    assert (out_dir / "dev_predictions.json").is_file()
    assert (out_dir / "e1_predictions.json").is_file()
    assert set(result["sets"].keys()) == {"dev", "e1"}
    assert result["sets"]["dev"]["n"] == 2
    assert "overall_ci" in result["sets"]["dev"]
    assert "length_buckets_e1_e2_e3" in result
    on_disk = json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))
    assert on_disk == result
