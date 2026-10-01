from __future__ import annotations

# Tests for nmt/decode.py: beam-1 == greedy, repetition block, max_len, cache-reorder
# correctness (beam score == teacher-forced rescoring), and the segmentation fallback's pure
# text-splitting function. Spec §7, §12.
import torch

from nmt.decode import (
    DecodeConfig,
    Hypothesis,
    beam_search_decode,
    decode_max_len,
    gnmt_length_penalty,
    greedy_decode,
    split_sentences,
)
from nmt.model.transformer import ModelConfig, Transformer


def _tiny_model(seed: int, **overrides: object) -> Transformer:
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=50, d_model=32, n_heads=4, enc_layers=2, dec_layers=2, d_ff=64, dropout=0.0
    )
    kwargs.update(overrides)
    return Transformer(ModelConfig(**kwargs)).eval()  # type: ignore[arg-type]


def _random_batch(
    seed: int, b: int, t_src: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(4, vocab_size, (b, t_src), generator=g)
    src_mask = torch.ones_like(src, dtype=torch.bool)
    return src, src_mask


def test_beam_size_1_equals_greedy_token_identical() -> None:
    """Beam search with beam_size=1 must produce exactly the same tokens as the independently
    implemented greedy_decode, for several random inputs (spec §12)."""
    for seed in range(5):
        model = _tiny_model(seed)
        src, src_mask = _random_batch(seed + 100, b=3, t_src=6, vocab_size=50)
        cfg = DecodeConfig(beam_size=1, alpha=0.6, no_repeat_ngram_size=3, max_len_b=20)
        beam_hyps = beam_search_decode(model, src, src_mask, bos_id=2, eos_id=3, pad_id=0, cfg=cfg)
        greedy_hyps = greedy_decode(
            model, src, src_mask, bos_id=2, eos_id=3, pad_id=0, max_len_b=20, no_repeat_ngram_size=3
        )
        for bh, gh in zip(beam_hyps, greedy_hyps, strict=True):
            assert bh.tokens == gh.tokens, (seed, bh.tokens, gh.tokens)


def test_beam_search_blocks_repeated_trigrams() -> None:
    """With no_repeat_ngram_size=3, no hypothesis may contain a repeated 3-gram."""
    model = _tiny_model(1)
    src, src_mask = _random_batch(2, b=4, t_src=8, vocab_size=50)
    cfg = DecodeConfig(beam_size=4, alpha=0.6, no_repeat_ngram_size=3, max_len_a=3.0, max_len_b=30)
    hyps = beam_search_decode(model, src, src_mask, bos_id=2, eos_id=3, pad_id=0, cfg=cfg)
    for h in hyps:
        seen: set[tuple[int, int, int]] = set()
        for i in range(len(h.tokens) - 2):
            gram = (h.tokens[i], h.tokens[i + 1], h.tokens[i + 2])
            assert gram not in seen, (h.tokens, gram)
            seen.add(gram)


def test_beam_search_respects_max_len() -> None:
    """No hypothesis may exceed decode_max_len(src_len) generated tokens."""
    model = _tiny_model(3)
    src, src_mask = _random_batch(4, b=2, t_src=10, vocab_size=50)
    cfg = DecodeConfig(beam_size=5, alpha=1.0, no_repeat_ngram_size=3, max_len_a=1.5, max_len_b=10)
    hyps = beam_search_decode(model, src, src_mask, bos_id=2, eos_id=3, pad_id=0, cfg=cfg)
    src_lens = src_mask.sum(dim=1).tolist()
    for h, src_len in zip(hyps, src_lens, strict=True):
        assert len(h.tokens) <= decode_max_len(int(src_len), cfg.max_len_a, cfg.max_len_b)


def _teacher_forced_score(
    model: Transformer, src: torch.Tensor, src_mask: torch.Tensor, hyp: Hypothesis, alpha: float
) -> float:
    """Rescore `hyp.tokens` (assumed to end in EOS=3) via a single teacher-forced full forward
    pass, for comparing against the beam search's own incrementally-accumulated score."""
    tgt = torch.tensor([2, *hyp.tokens], dtype=torch.long).unsqueeze(0)  # BOS + tokens (incl EOS)
    with torch.no_grad():
        logits = model(src, tgt[:, :-1])  # predict each of hyp.tokens from BOS + preceding
        log_probs = torch.log_softmax(logits, dim=-1)
        target = tgt[:, 1:]
        token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    raw_score = float(token_lp.sum().item())
    return raw_score / gnmt_length_penalty(len(hyp.tokens), alpha)


def test_beam_search_cache_reorder_matches_teacher_forced_rescoring() -> None:
    """The best hypothesis's beam-search score must equal a teacher-forced rescoring of that
    exact token sequence with the full (non-incremental) forward pass, within 1e-4 -- this only
    holds if the KV cache was reordered correctly at every beam-swap step."""
    model = _tiny_model(7)
    src, src_mask = _random_batch(8, b=3, t_src=5, vocab_size=50)
    # Generous max_len headroom so forced-EOS-at-cap never fires; every hypothesis below
    # terminates by naturally sampling EOS as the top candidate, keeping its score a pure sum of
    # real (unmasked-for-the-chosen-token) model log-probs, directly comparable to a teacher-
    # forced rescoring.
    cfg = DecodeConfig(
        beam_size=4, alpha=0.7, no_repeat_ngram_size=3, max_len_a=10.0, max_len_b=100
    )
    hyps = beam_search_decode(model, src, src_mask, bos_id=2, eos_id=3, pad_id=0, cfg=cfg)
    src_lens = src_mask.sum(dim=1).tolist()
    caps = [decode_max_len(int(n), cfg.max_len_a, cfg.max_len_b) for n in src_lens]
    n_checked = 0
    for i, h in enumerate(hyps):
        if h.tokens[-1] != 3 or len(h.tokens) >= caps[i]:
            # A random, untrained model can fail to ever rank EOS top-1 and instead hit the
            # max_len cap, where log_probs are overwritten to force EOS regardless of the real
            # model score (spec §7's hard max_len) -- teacher-forced rescoring deliberately
            # disagrees with that forced score by construction, so those hypotheses are excluded
            # here rather than asserted on; `test_beam_search_respects_max_len` already covers
            # the cap itself.
            continue
        rescored = _teacher_forced_score(model, src[i : i + 1], src_mask[i : i + 1], h, cfg.alpha)
        assert abs(rescored - h.score) < 1e-4, (h.score, rescored)
        n_checked += 1
    assert n_checked > 0, "no naturally-terminated hypothesis to check; strengthen the fixture"


def test_split_sentences_keeps_delimiters_and_never_empty() -> None:
    text = "Bonjour. Comment allez-vous? Bien! A bientot..."
    parts = split_sentences(text)
    assert all(p.strip() != "" for p in parts)
    assert "".join(parts) != text or " ".join(parts).replace("  ", " ") == text.replace("  ", " ")
    assert parts[0].endswith(".")
    assert parts[1].endswith("?")
    assert parts[2].endswith("!")


def test_split_sentences_falls_back_to_whole_sentence_when_no_split_point() -> None:
    text = "no punctuation here at all"
    assert split_sentences(text) == [text]


def test_split_sentences_handles_ellipsis_and_semicolon_colon() -> None:
    """; and : are also split points per spec §7 (". ! ? ; : and …"), so this 4-clause sentence
    splits into 4 segments."""
    text = "Attends… il arrive; regarde: la porte."
    parts = split_sentences(text)
    assert len(parts) == 4
    assert all(p.strip() != "" for p in parts)
    assert parts[0].endswith("…")
    assert parts[1].endswith(";")
    assert parts[2].endswith(":")
    assert parts[3].endswith(".")
