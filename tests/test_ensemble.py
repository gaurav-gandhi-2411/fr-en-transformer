from __future__ import annotations

# nmt/ensemble.py: log-prob averaging, parity with the single model, tokenizer/vocab refusal.
import copy
from pathlib import Path

import pytest
import torch

from nmt.decode import DecodeConfig, beam_search_decode, greedy_decode
from nmt.ensemble import Ensemble, EnsembleError, check_compatible
from nmt.hub import NMTModel, export_checkpoint
from nmt.mbr import MBRConfig, beam_pool, sample_pool
from nmt.model.transformer import ModelConfig, Transformer
from nmt.translate import Translator


def _tiny(seed: int, vocab: int = 48) -> NMTModel:
    torch.manual_seed(seed)
    return NMTModel(
        vocab_size=vocab, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32
    ).eval()


def _src() -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.tensor([[5, 6, 7, 3], [8, 9, 3, 0], [4, 3, 0, 0]])
    return src, src != 0


def _same(a: list, b: list) -> bool:
    return [(h.tokens, h.score) for h in a] == [(h.tokens, h.score) for h in b]


@pytest.mark.parametrize("beam", [1, 4])
def test_ensemble_of_one_equals_single_model_exactly(beam: int) -> None:
    m = _tiny(0)
    src, mask = _src()
    cfg = DecodeConfig(beam_size=beam, alpha=1.2)
    single = beam_search_decode(m, src, mask, 2, 3, 0, cfg)
    ens = beam_search_decode(Ensemble([m]), src, mask, 2, 3, 0, cfg)
    assert _same(single, ens)


def test_ensemble_of_two_identical_models_equals_single() -> None:
    m = _tiny(1)
    twin = copy.deepcopy(m)
    src, mask = _src()
    cfg = DecodeConfig(beam_size=4, alpha=1.0)
    assert _same(
        beam_search_decode(m, src, mask, 2, 3, 0, cfg),
        beam_search_decode(Ensemble([m, twin]), src, mask, 2, 3, 0, cfg),
    )
    assert _same(
        greedy_decode(m, src, mask, 2, 3, 0),
        greedy_decode(Ensemble([m, twin]), src, mask, 2, 3, 0),
    )


def test_step_is_mean_of_member_log_probs() -> None:
    a, b = _tiny(2), _tiny(3)
    src, mask = _src()
    ens = Ensemble([a, b])
    tok = torch.full((3, 1), 2, dtype=torch.long)
    with torch.no_grad():
        cache = ens.init_decode_cache(ens.encode(src, mask), mask)
        got = ens.decode_step(tok, cache)[:, 0, :]
        want = []
        for m in (a, b):
            c = m.init_decode_cache(m.encode(src, mask), mask)
            want.append(torch.log_softmax(m.decode_step(tok, c)[:, 0, :], dim=-1))
    manual = (want[0] + want[1]) / 2
    assert torch.allclose(got, manual, atol=1e-6)
    assert not torch.allclose(got, want[0])  # distinct members really differ
    # the mean of log-probs is NOT renormalized (a geometric mean sums to < 1 in probability)
    assert (got.exp().sum(dim=-1) < 1.0).all()


def test_two_different_models_run_beam_search() -> None:
    a, b = _tiny(4), _tiny(5)
    src, mask = _src()
    hyps = beam_search_decode(Ensemble([a, b]), src, mask, 2, 3, 0, DecodeConfig(beam_size=3))
    assert len(hyps) == 3 and all(h.tokens for h in hyps)


def test_incompatible_vocab_refused() -> None:
    with pytest.raises(EnsembleError, match="vocab_size"):
        Ensemble([_tiny(0, 48), _tiny(1, 64)])


def test_different_tokenizer_sha_refused() -> None:
    a, b = _tiny(0), _tiny(1)
    with pytest.raises(EnsembleError, match="different SentencePiece"):
        Ensemble([a, b], ["aa", "bb"])
    with pytest.raises(EnsembleError, match="missing"):
        check_compatible([a.config, b.config], ["aa", None])
    Ensemble([a, b], ["aa", "aa"])  # same tokenizer: accepted


def test_empty_ensemble_refused() -> None:
    with pytest.raises(EnsembleError):
        check_compatible([])


# ---- Translator integration (real tokenizer, tiny random models) ----

REAL_TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "spm.model"
_needs_tok = pytest.mark.skipif(not REAL_TOKENIZER.is_file(), reason="tokenizer not built")
_TEXTS = ["Bonjour le monde.", "Je ne sais pas.", "Merci beaucoup pour votre aide aujourd'hui."]


def _export(tmp_path: Path, name: str, seed: int) -> Path:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=16000, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32)
    ckpt = tmp_path / f"{name}.pt"
    torch.save({"step": 1, "model": Transformer(cfg).state_dict()}, ckpt)
    return export_checkpoint([ckpt], tmp_path / name, cfg, REAL_TOKENIZER, average=True)


@_needs_tok
def test_ensemble_translator_of_one_matches_plain_translator(tmp_path: Path) -> None:
    from nmt.ensemble import load_ensemble_translator

    d = _export(tmp_path, "m0", 0)
    plain = Translator.from_pretrained(str(d), device="cpu")
    ens = load_ensemble_translator([d], device="cpu")
    for kw in ({"beam": 1}, {"beam": 4, "alpha": 1.4}):
        assert plain.translate(_TEXTS, **kw) == ens.translate(_TEXTS, **kw)


@_needs_tok
def test_ensemble_translator_mbr_and_refusal(tmp_path: Path) -> None:
    from nmt.ensemble import load_ensemble_translator

    a, b = _export(tmp_path, "a", 1), _export(tmp_path, "b", 2)
    ens = load_ensemble_translator([a, b], device="cpu")
    for mbr in (MBRConfig("beam", 4), MBRConfig("sample", 4)):
        out = ens.translate(_TEXTS, alpha=1.2, mbr=mbr)
        assert len(out) == 3 and all(o.strip() for o in out)
        assert out == ens.translate(_TEXTS, alpha=1.2, mbr=mbr)  # deterministic
    # a member exported with a different tokenizer file is refused
    other = _export(tmp_path, "c", 3)
    with (other / "spm.model").open("ab") as f:
        f.write(b"\x00")
    with pytest.raises(EnsembleError, match="different SentencePiece"):
        load_ensemble_translator([a, other], device="cpu")


def test_members_are_in_eval_mode_after_construction_and_after_decode_helpers() -> None:
    # members built by _tiny are eval(); force train mode first so only Ensemble.__init__ can fix it
    a, b = _tiny(6).train(), _tiny(7).train()
    ens = Ensemble([a, b])
    assert not ens.training and not a.training and not b.training
    src, mask = _src()
    beam_pool(ens, src, mask, 2, 3, 0, n=3, alpha=1.0)
    assert not ens.training and not a.training and not b.training
    sample_pool(ens, src, mask, 2, 3, ["a", "b", "c"], n=3)
    assert not ens.training and not a.training and not b.training
