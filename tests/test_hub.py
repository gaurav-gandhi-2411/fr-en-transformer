from __future__ import annotations

# Tests for nmt/hub.py: export_checkpoint -> save_pretrained -> load_pretrained round-trips to
# byte-identical logits, single vs averaged checkpoints differ predictably, and the tokenizer
# file travels alongside the export. Spec §13.
from pathlib import Path

import torch

from nmt.hub import NMTModel, export_checkpoint, load_pretrained
from nmt.model.transformer import ModelConfig, Transformer

_CFG_KWARGS = dict(
    vocab_size=50, d_model=32, n_heads=4, enc_layers=2, dec_layers=2, d_ff=64, dropout=0.0
)


def _make_checkpoint(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = Transformer(ModelConfig(**_CFG_KWARGS))  # type: ignore[arg-type]
    torch.save({"step": seed, "model": model.state_dict()}, path)


def test_export_then_load_round_trips_to_identical_logits(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "step_00000010.pt"
    _make_checkpoint(ckpt_path, seed=0)
    tok_path = tmp_path / "spm.model"
    tok_path.write_bytes(b"fake-spm-bytes-for-round-trip-test")

    cfg = ModelConfig(**_CFG_KWARGS)  # type: ignore[arg-type]
    out_dir = export_checkpoint([ckpt_path], tmp_path / "export", cfg, tok_path, average=True)

    assert (out_dir / "config.json").is_file()
    assert (out_dir / "model.safetensors").is_file()
    assert (out_dir / "spm.model").read_bytes() == b"fake-spm-bytes-for-round-trip-test"

    loaded, spm_path = load_pretrained(str(out_dir))
    assert isinstance(loaded, NMTModel)
    assert spm_path == out_dir / "spm.model"
    assert loaded.config == cfg

    original_state = torch.load(ckpt_path, weights_only=False)["model"]
    reference = Transformer(cfg)
    reference.load_state_dict(original_state)

    src = torch.randint(4, 50, (2, 6))
    tgt_in = torch.randint(4, 50, (2, 5))
    with torch.no_grad():
        expected = reference(src, tgt_in)
        actual = loaded(src, tgt_in)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_averaged_export_differs_from_either_single_checkpoint(tmp_path: Path) -> None:
    """average=True over two distinct checkpoints must not equal either one alone (spec §6:
    "report averaged vs single" implies they are, in general, different models)."""
    ckpt_a = tmp_path / "step_00000010.pt"
    ckpt_b = tmp_path / "step_00000020.pt"
    _make_checkpoint(ckpt_a, seed=1)
    _make_checkpoint(ckpt_b, seed=2)
    tok_path = tmp_path / "spm.model"
    tok_path.write_bytes(b"x")
    cfg = ModelConfig(**_CFG_KWARGS)  # type: ignore[arg-type]

    averaged_dir = export_checkpoint(
        [ckpt_a, ckpt_b], tmp_path / "avg", cfg, tok_path, average=True
    )
    single_dir = export_checkpoint(
        [ckpt_a, ckpt_b], tmp_path / "single", cfg, tok_path, average=False
    )

    averaged_model, _ = load_pretrained(str(averaged_dir))
    single_model, _ = load_pretrained(str(single_dir))

    # average=False must use ckpt_paths[-1] (ckpt_b), byte-identical to loading ckpt_b directly.
    expected_single = Transformer(cfg)
    expected_single.load_state_dict(torch.load(ckpt_b, weights_only=False)["model"])
    src = torch.randint(4, 50, (2, 6))
    tgt_in = torch.randint(4, 50, (2, 5))
    with torch.no_grad():
        single_out = single_model(src, tgt_in)
        expected_single_out = expected_single(src, tgt_in)
        averaged_out = averaged_model(src, tgt_in)
    assert torch.allclose(single_out, expected_single_out, atol=1e-6)
    assert not torch.allclose(single_out, averaged_out, atol=1e-4)
