from __future__ import annotations

# Tests for nmt/model/transformer.py: padding/causal masks, RoPE's relative property, KV-cache
# incremental decoding vs full-sequence decoding, and a tiny-model-overfits-one-batch sanity
# check. Spec §12.
import torch

from nmt.model.transformer import ModelConfig, Transformer, apply_rope
from nmt.train import label_smoothed_nll_loss


def _tiny_config(**overrides: object) -> ModelConfig:
    kwargs = dict(
        vocab_size=50, d_model=32, n_heads=4, enc_layers=2, dec_layers=2, d_ff=64, dropout=0.0
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)  # type: ignore[arg-type]


def test_padding_mask_ignores_padded_keys_and_leaves_valid_outputs_unchanged() -> None:
    """Extending the source with extra pad tokens must not change the encoder output at the
    original (non-pad) positions: those positions never attend to the new pad keys (masked out),
    and every other op (LayerNorm, FFN, RoPE-by-absolute-position) is per-token.
    """
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Transformer(cfg).eval()

    src = torch.randint(4, cfg.vocab_size, (2, 6))
    src_mask = torch.ones_like(src, dtype=torch.bool)
    memory_short = model.encode(src, src_mask)

    pad_id = cfg.pad_id
    extra_pad = torch.full((2, 5), pad_id, dtype=torch.long)
    src_padded = torch.cat([src, extra_pad], dim=1)
    src_padded_mask = src_padded != pad_id
    memory_padded = model.encode(src_padded, src_padded_mask)

    assert torch.allclose(memory_short, memory_padded[:, :6, :], atol=1e-6)


def test_causal_mask_blocks_future_leakage() -> None:
    """Perturbing a future target token must leave logits at earlier positions unchanged
    (exact equality in eval mode, dropout off, no randomness in the forward pass).
    """
    torch.manual_seed(1)
    cfg = _tiny_config()
    model = Transformer(cfg).eval()

    src = torch.randint(4, cfg.vocab_size, (2, 6))
    tgt_in = torch.randint(4, cfg.vocab_size, (2, 5))

    with torch.no_grad():
        logits_a = model(src, tgt_in)
        tgt_in_perturbed = tgt_in.clone()
        tgt_in_perturbed[:, -1] = (tgt_in_perturbed[:, -1] + 7) % cfg.vocab_size
        logits_b = model(src, tgt_in_perturbed)

    # Every position except the last (whose input token was changed) must be identical.
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-6)
    assert not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :])


def test_rope_relative_property_depends_only_on_offset() -> None:
    """q_m . k_n after RoPE depends only on (m - n): shifting both positions by the same offset
    s leaves the dot product unchanged (Su et al. 2021).
    """
    torch.manual_seed(2)
    head_dim = 8
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    q = torch.randn(5, head_dim)
    k = torch.randn(5, head_dim)

    m = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
    n = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
    s = 17.0

    q_rot = apply_rope(q, m, inv_freq)
    k_rot = apply_rope(k, n, inv_freq)
    dots = (q_rot * k_rot).sum(dim=-1)

    q_rot_shifted = apply_rope(q, m + s, inv_freq)
    k_rot_shifted = apply_rope(k, n + s, inv_freq)
    dots_shifted = (q_rot_shifted * k_rot_shifted).sum(dim=-1)

    assert torch.allclose(dots, dots_shifted, atol=1e-4)


def test_kv_cache_incremental_decoding_matches_full_sequence() -> None:
    """Decoding one token at a time through the KV cache must reproduce the teacher-forced,
    full-sequence `decode` logits exactly (up to float tolerance).
    """
    torch.manual_seed(3)
    cfg = _tiny_config()
    model = Transformer(cfg).eval()

    src = torch.randint(4, cfg.vocab_size, (2, 7))
    tgt_in = torch.randint(4, cfg.vocab_size, (2, 6))
    src_mask = src != cfg.pad_id

    with torch.no_grad():
        full_logits = model(src, tgt_in)

        memory = model.encode(src, src_mask)
        cache = model.init_decode_cache(memory, src_mask)
        step_logits = []
        for t in range(tgt_in.size(1)):
            step_logits.append(model.decode_step(tgt_in[:, t : t + 1], cache))
        incremental_logits = torch.cat(step_logits, dim=1)

    assert torch.allclose(full_logits, incremental_logits, atol=1e-5)


def test_rope_and_sinusoidal_configs_both_produce_finite_logits() -> None:
    """Smoke-check the `pos` config switch (spec §5/§9 ablation) end to end for both settings."""
    torch.manual_seed(4)
    src = torch.randint(4, 50, (2, 6))
    tgt_in = torch.randint(4, 50, (2, 5))
    for pos in ("rope", "sinusoidal"):
        model = Transformer(_tiny_config(pos=pos)).eval()
        with torch.no_grad():
            logits = model(src, tgt_in)
        assert torch.isfinite(logits).all()


def test_sinusoidal_positions_beyond_max_len_do_not_crash() -> None:
    """decode.py's max_len = floor(1.5*src_len) + 10 can exceed cfg.max_len on a long source;
    the sinusoidal table must grow to cover it instead of an index-out-of-bounds crash.
    """
    torch.manual_seed(6)
    cfg = _tiny_config(pos="sinusoidal", max_len=16)  # table starts far smaller than 600
    model = Transformer(cfg).eval()

    src = torch.randint(4, cfg.vocab_size, (1, 5))
    src_mask = torch.ones_like(src, dtype=torch.bool)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        cache = model.init_decode_cache(memory, src_mask)
        token = torch.randint(4, cfg.vocab_size, (1, 1))
        logits = None
        for _ in range(600):
            logits = model.decode_step(token, cache)
            assert torch.isfinite(logits).all()
    assert model.pos_table.size(0) >= 600
    assert logits is not None


def test_tiny_model_overfits_one_batch() -> None:
    """A tiny model trained on a single fixed batch with plain CE (epsilon=0 — label smoothing's
    floor above 0 would make a hard 0.1 threshold arbitrary/unreachable by construction) should
    drive the loss below 0.1 within a modest number of steps.
    """
    torch.manual_seed(5)
    cfg = _tiny_config(dropout=0.0)
    model = Transformer(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    src = torch.randint(4, cfg.vocab_size, (4, 6))
    tgt_in = torch.randint(4, cfg.vocab_size, (4, 5))
    tgt_out = torch.randint(4, cfg.vocab_size, (4, 5))

    loss = torch.tensor(float("inf"))
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt_in)
        loss = label_smoothed_nll_loss(logits, tgt_out, pad_id=cfg.pad_id, epsilon=0.0)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1
