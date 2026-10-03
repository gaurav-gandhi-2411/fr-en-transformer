from __future__ import annotations

# Hugging Face Hub export/load via PyTorchModelHubMixin (safetensors, config.json, tokenizer)
# and a model-card render stub.
#
# `NMTModel` wraps `nmt.model.transformer.Transformer` so `save_pretrained(dir)` writes
# `model.safetensors` + `config.json` (no network call unless `push_to_hub=True`) and
# `from_pretrained(dir)` reconstructs the exact `ModelConfig` from `config.json` before loading
# weights. The tokenizer's `spm.model` is copied alongside on export/load so both travel
# together as one directory/repo.
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch import Tensor, nn

from nmt.checkpoint import average_checkpoints
from nmt.model.transformer import ModelConfig, Transformer

TOKENIZER_FILENAME = "spm.model"


class NMTModel(nn.Module, PyTorchModelHubMixin):
    """`PyTorchModelHubMixin` captures every `__init__` keyword argument as `config.json`
    (`ModelHubMixin`'s constructor-wrapping behavior), so this signature IS the exported config
    schema -- it must stay a flat, JSON-serializable mirror of `ModelConfig`'s fields.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 512,
        n_heads: int = 8,
        enc_layers: int = 8,
        dec_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pos: str = "rope",
        max_len: int = 512,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            d_ff=d_ff,
            dropout=dropout,
            pos=pos,
            max_len=max_len,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            rope_base=rope_base,
        )
        self.transformer = Transformer(self.config)

    # -- thin delegation so callers (Translator, evaluate.py) can treat an NMTModel exactly like
    # the Transformer it wraps, without reaching into `.transformer` at every call site. --
    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        return self.transformer(src, tgt_in)

    def encode(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        return self.transformer.encode(src, src_key_padding_mask)

    def init_decode_cache(self, memory: Tensor, src_key_padding_mask: Tensor) -> dict:
        return self.transformer.init_decode_cache(memory, src_key_padding_mask)

    def decode_step(self, tgt_token: Tensor, cache: dict) -> Tensor:
        return self.transformer.decode_step(tgt_token, cache)

    def param_count(self) -> int:
        return self.transformer.param_count()


def export_checkpoint(
    ckpt_paths: Sequence[Path],
    out_dir: Path,
    model_cfg: ModelConfig,
    tokenizer_path: Path,
    average: bool = True,
) -> Path:
    """Build an `NMTModel` from one or more `nmt.train.save_checkpoint` files and `save_pretrained`
    it to `out_dir`, with the tokenizer copied alongside.

    `average=True` (default): element-wise mean of every checkpoint's weights via
    `nmt.checkpoint.average_checkpoints` -- with a single-element `ckpt_paths` this is just that
    checkpoint's own weights (mean of one). `average=False`: only `ckpt_paths[-1]`'s raw weights
    are used, so callers comparing "averaged vs single" pass the same checkpoint list
    both ways and vary only this flag.
    """
    if average:
        state = average_checkpoints(list(ckpt_paths))
    else:
        payload = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=False)
        state = payload["model"]

    hub_model = NMTModel(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        enc_layers=model_cfg.enc_layers,
        dec_layers=model_cfg.dec_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        pos=model_cfg.pos,
        max_len=model_cfg.max_len,
        pad_id=model_cfg.pad_id,
        bos_id=model_cfg.bos_id,
        eos_id=model_cfg.eos_id,
        rope_base=model_cfg.rope_base,
    )
    hub_model.transformer.load_state_dict(state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_model.save_pretrained(out_dir)
    shutil.copy2(Path(tokenizer_path), out_dir / TOKENIZER_FILENAME)
    return out_dir


def load_pretrained(path_or_repo_id: str) -> tuple[NMTModel, Path]:
    """Load an `NMTModel` (weights + config) plus resolve its tokenizer's `spm.model` path --
    from a local export directory (`nmt.hub.export_checkpoint`'s output) or an HF Hub repo id,
    through the same `PyTorchModelHubMixin.from_pretrained` codepath either way. Never pushes
    anything (read-only); the caller decides whether `path_or_repo_id` is local or remote by what
    it passes, exactly like every other `from_pretrained` in the ecosystem.
    """
    model = NMTModel.from_pretrained(path_or_repo_id)
    local_dir = Path(path_or_repo_id)
    if local_dir.is_dir():
        spm_path = local_dir / TOKENIZER_FILENAME
    else:
        spm_path = Path(hf_hub_download(repo_id=path_or_repo_id, filename=TOKENIZER_FILENAME))
    return model, spm_path


def render_model_card(*_args: object, **_kwargs: object) -> str:
    """Stub: model-card rendering is not implemented."""
    raise NotImplementedError("model card: see report/hf_model_card.md")
