from __future__ import annotations

# Tests for nmt/translate.py: from_pretrained + translate() round-trip against the real
# tokenizer, order-preservation under length-sorted batching, the never-empty-output guarantee
# (including a pathological empty-string input), segmentation-fallback join behavior, and
# fallback-count bookkeeping. Spec §7, §12.
from pathlib import Path

import pytest
import torch

from nmt.hub import export_checkpoint
from nmt.model.transformer import ModelConfig
from nmt.translate import Translator, benchmark_translator, model_size_mb

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TOKENIZER = REPO_ROOT / "tokenizer" / "spm.model"

pytestmark = pytest.mark.skipif(
    not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)"
)


def _export_tiny_model(tmp_path: Path, seed: int = 0) -> Path:
    """Export a tiny (real-vocab-sized, small d_model) model + the real tokenizer, matching
    `nmt.hub.export_checkpoint`'s directory contract, for a fast but real translate() round-trip.
    """
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=16000, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32)
    from nmt.model.transformer import Transformer

    model = Transformer(cfg)
    ckpt_path = tmp_path / "step_00000010.pt"
    torch.save({"step": 10, "model": model.state_dict()}, ckpt_path)
    out_dir = tmp_path / "export"
    export_checkpoint([ckpt_path], out_dir, cfg, REAL_TOKENIZER, average=True)
    return out_dir


def test_translate_returns_nonempty_strings_in_input_order(tmp_path: Path) -> None:
    export_dir = _export_tiny_model(tmp_path)
    translator = Translator.from_pretrained(str(export_dir), device="cpu")
    texts = [
        "Bonjour le monde.",
        "Je ne sais pas.",  # short
        "Ceci est une phrase un peu plus longue pour tester le tri par longueur avant le batching.",
        "Merci.",
    ]
    outputs = translator.translate(texts, batch_size=2, beam=2, alpha=0.6)
    assert len(outputs) == len(texts)
    assert all(isinstance(o, str) and o.strip() != "" for o in outputs)
    assert translator.stats.n_total == len(texts)  # no segmentation: 1 work item per input


def test_translate_never_emits_empty_string_for_empty_input(tmp_path: Path) -> None:
    export_dir = _export_tiny_model(tmp_path, seed=1)
    translator = Translator.from_pretrained(str(export_dir), device="cpu")
    outputs = translator.translate(["", "   ", "Bonjour."], batch_size=8, beam=1, alpha=0.6)
    assert all(o != "" for o in outputs)


def test_translate_with_segmentation_threshold_joins_with_space(tmp_path: Path) -> None:
    export_dir = _export_tiny_model(tmp_path, seed=2)
    translator = Translator.from_pretrained(str(export_dir), device="cpu")
    text = "Bonjour. Comment allez-vous? Tres bien, merci!"
    outputs = translator.translate([text], batch_size=8, beam=1, alpha=0.6, segment_threshold=1)
    assert len(outputs) == 1
    assert outputs[0].strip() != ""
    # threshold=1 forces every multi-token source to segment; 3 sentence-like work items.
    assert translator.stats.n_total == 3


def test_fallback_counts_are_consistent_with_stats(tmp_path: Path) -> None:
    export_dir = _export_tiny_model(tmp_path, seed=3)
    translator = Translator.from_pretrained(str(export_dir), device="cpu")
    translator.translate(["Bonjour.", "Au revoir."], batch_size=2, beam=2, alpha=0.6)
    total = (
        translator.stats.n_beam
        + translator.stats.n_greedy_fallback
        + translator.stats.n_copy_fallback
    )
    assert total == translator.stats.n_total


def test_benchmark_translator_reports_positive_throughput_and_model_size(tmp_path: Path) -> None:
    export_dir = _export_tiny_model(tmp_path, seed=4)
    translator = Translator.from_pretrained(str(export_dir), device="cpu")
    texts = ["Bonjour.", "Au revoir.", "Merci beaucoup.", "A bientot."]
    result = benchmark_translator(translator, texts, batch_size=2, beam=1, alpha=0.6)
    assert result.n_sentences == len(texts)
    assert result.sentences_per_second > 0
    assert result.batch_latency_p50_ms >= 0
    assert result.batch_latency_p95_ms >= result.batch_latency_p50_ms - 1e-6
    assert result.model_size_mb == pytest.approx(model_size_mb(translator.model))
    assert result.memory_metric in {"tracemalloc_python_peak_mb", "cuda_max_memory_allocated_mb"}
