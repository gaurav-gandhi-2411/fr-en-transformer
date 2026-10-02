from __future__ import annotations

# Ensemble decoding (PREREG post-selection amendment, rule 3): average the per-step
# log-probabilities of several models that share one SentencePiece tokenizer. `Ensemble` exposes
# the same decoder-step interface as `nmt.hub.NMTModel` (encode / init_decode_cache / decode_step /
# config), so `nmt.decode.beam_search_decode`, `greedy_decode` and `nmt.mbr.sample_pool` and the
# `nmt.translate.Translator` use it unchanged apart from two tiny hooks in nmt/decode.py (the
# step returns log-probs, and the encoder memory / cache are per member).
#
# The combination is the arithmetic mean of the members' log-softmax outputs (a log-linear /
# geometric-mean ensemble), NOT renormalized: PREREG says "average per-step log-probabilities",
# and a renormalized (probability-space) mean would be a different rule. For beam search the
# constant per-step offset a missing renormalization introduces is common to all tokens of a beam
# at a step, but not across beams, so it is part of the rule, not a bug. Samplers that need a
# proper distribution renormalize (nmt.mbr.sample_pool does, via log_softmax).
import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from nmt.hub import TOKENIZER_FILENAME, NMTModel, load_pretrained
from nmt.translate import Translator


class EnsembleError(ValueError):
    """The members cannot be ensembled (different tokenizer / vocabulary / special ids)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_compatible(
    configs: Sequence[object], tokenizer_sha256s: Sequence[str | None] | None = None
) -> None:
    """Raise EnsembleError unless every member has the same vocab size and special-token ids and
    (when given) the same tokenizer sha256. Fail closed: a missing (None) sha256 is a refusal."""
    if not configs:
        raise EnsembleError("an ensemble needs at least one model")
    first = configs[0]
    for cfg in configs[1:]:
        for field in ("vocab_size", "pad_id", "bos_id", "eos_id"):
            if getattr(cfg, field) != getattr(first, field):
                raise EnsembleError(
                    f"ensemble members disagree on {field}: "
                    f"{getattr(first, field)} vs {getattr(cfg, field)}"
                )
    if tokenizer_sha256s is not None:
        if len(tokenizer_sha256s) != len(configs) or any(s is None for s in tokenizer_sha256s):
            raise EnsembleError("tokenizer sha256 missing for at least one ensemble member")
        if len(set(tokenizer_sha256s)) != 1:
            raise EnsembleError(
                "ensemble members use different SentencePiece tokenizers (sha256 "
                f"{sorted(set(tokenizer_sha256s))})"
            )


class Ensemble(nn.Module):
    """Average of per-step log-probabilities over `models` (see module docstring)."""

    # Read by nmt.decode._step_log_probs: decode_step already returns log-probs.
    step_returns_log_probs = True

    def __init__(
        self, models: Sequence[NMTModel], tokenizer_sha256s: Sequence[str | None] | None = None
    ) -> None:
        super().__init__()
        models = list(models)
        check_compatible([m.config for m in models], tokenizer_sha256s)
        self.members = nn.ModuleList(models)
        self.config = models[0].config
        self.eval()  # decoding only: a freshly built nn.Module is in train mode (dropout on)

    def encode(self, src: Tensor, src_key_padding_mask: Tensor) -> tuple[Tensor, ...]:
        return tuple(m.encode(src, src_key_padding_mask) for m in self.members)

    def init_decode_cache(
        self, memory: tuple[Tensor, ...], src_key_padding_mask: Tensor
    ) -> dict[str, list[dict]]:
        """One independent KV cache per member (the beam search reorders each of them)."""
        return {
            "members": [
                m.init_decode_cache(mem, src_key_padding_mask)
                for m, mem in zip(self.members, memory, strict=True)
            ]
        }

    def decode_step(self, tgt_token: Tensor, cache: dict) -> Tensor:
        """Mean of the members' log_softmax outputs, shape (B, 1, V). The mean of one member is
        exactly that member's own log_softmax (division by 1), so an ensemble of 1 reproduces the
        single model bit for bit."""
        per_member = [
            torch.log_softmax(m.decode_step(tgt_token, c), dim=-1)
            for m, c in zip(self.members, cache["members"], strict=True)
        ]
        if len(per_member) == 1:
            return per_member[0]
        return torch.stack(per_member, dim=0).mean(dim=0)


def load_ensemble_translator(
    model_dirs: Sequence[str | Path], device: str | torch.device | None = None
) -> Translator:
    """A `Translator` over the ensemble of the HF-style export dirs `model_dirs` (one dir = the
    plain single-model translator's model). Refuses members whose spm.model sha256, vocab size or
    special ids differ. Read-only."""
    if not model_dirs:
        raise EnsembleError("an ensemble needs at least one model dir")
    models, spm_paths = [], []
    for d in model_dirs:
        model, spm_path = load_pretrained(str(d))
        models.append(model)
        spm_paths.append(spm_path)
    assert all(p.name == TOKENIZER_FILENAME for p in spm_paths)
    ensemble = Ensemble(models, [_sha256_file(p) for p in spm_paths])
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_paths[0]))
    resolved = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return Translator(ensemble, sp, resolved)  # type: ignore[arg-type]
