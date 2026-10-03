from __future__ import annotations

# Hand-written encoder-decoder transformer: 8 encoder / 4 decoder layers, d=512,
# 8 heads, FFN 2048, pre-LayerNorm, three-way tied embeddings, RoPE/sinusoidal
# positional-encoding switch, F.scaled_dot_product_attention, decoder KV cache for
# incremental decoding.
#
# No nn.Transformer / nn.TransformerEncoderLayer / nn.MultiheadAttention anywhere in this file
# (the point of a from-scratch implementation): every linear projection, mask and positional
# encoding is written out explicitly so it can be read line by line.
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    """Architecture + special-token config. `vocab_size` comes from the trained SentencePiece
    model (special ids: pad=0, unk=1, bos=2, eos=3); everything else is a hyperparameter with
    the main-run defaults.
    """

    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 8
    dec_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    pos: str = "rope"  # "rope" | "sinusoidal" — ablation switch
    max_len: int = 512  # sinusoidal table size and an implicit cap on positions seen at train time
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.pos not in ("rope", "sinusoidal"):
            raise ValueError(f"pos must be 'rope' or 'sinusoidal', got {self.pos!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


def _sinusoidal_table(max_len: int, d_model: int) -> Tensor:
    """Standard Vaswani et al. 2017 sinusoidal position table, shape (max_len, d_model)."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table


def apply_rope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    """Rotary position embedding (Su et al. 2021), rotate-half convention.

    `x` is (..., T, Dh) with the same T as `positions` (shape (T,), shared across every leading
    dim — encoder/decoder self-attention always applies the same positions across the whole
    batch, so a per-batch position tensor is unnecessary here). `inv_freq` is (Dh/2,).

    This construction gives the RoPE relative property exactly: for any offset s,
    rope(x, m) . rope(y, n) == rope(x, m+s) . rope(y, n+s), because rotating both operands by
    the same extra angle (s * inv_freq) leaves their relative angle, and hence their dot product,
    unchanged. Tested directly in tests/test_model.py.
    """
    freqs = positions.to(inv_freq.dtype)[:, None] * inv_freq[None, :]  # (T, Dh/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, Dh)
    cos = emb.cos().to(dtype=x.dtype, device=x.device)
    sin = emb.sin().to(dtype=x.dtype, device=x.device)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """One multi-head attention block backed by F.scaled_dot_product_attention.

    K/V projection and Q projection/attention are split into two methods (`project_kv` and
    `forward`) so that incremental decoding can project+rotate only the *new* token's K/V,
    concatenate with an already-rotated cache, and attend against the full cached sequence —
    RoPE must be applied to each token at the absolute position it was produced, before caching,
    not re-derived after concatenation.
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float, use_rope: bool, rope_base: float
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        if use_rope:
            exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            inv_freq = 1.0 / (rope_base**exponent)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def project_kv(self, kv_in: Tensor, positions: Tensor | None) -> tuple[Tensor, Tensor]:
        """Project (and, for self-attention, rotate) K/V from `kv_in`. `positions` is None for
        cross-attention (no positional term on cross-attention).
        """
        k = self._split_heads(self.k_proj(kv_in))
        v = self._split_heads(self.v_proj(kv_in))
        if self.use_rope and positions is not None:
            k = apply_rope(k, positions, self.inv_freq)
        return k, v

    def forward(
        self,
        q_in: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
        q_positions: Tensor | None,
        training: bool,
    ) -> Tensor:
        """`k`/`v` are already-projected (and, if applicable, RoPE-rotated) tensors of shape
        (B,H,T_k,Dh) — produced by `project_kv`, possibly concatenated with a KV cache by the
        caller. `q_in` is the raw (pre-projection) query input of shape (B,T_q,D).
        """
        q = self._split_heads(self.q_proj(q_in))
        if self.use_rope and q_positions is not None:
            q = apply_rope(q, q_positions, self.inv_freq)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if training else 0.0
        )
        return self.out_proj(self._merge_heads(out))


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Pre-LN encoder block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.dropout,
            use_rope=(cfg.pos == "rope"),
            rope_base=cfg.rope_base,
        )
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None, positions: Tensor) -> Tensor:
        h = self.norm1(x)
        k, v = self.self_attn.project_kv(h, positions)
        x = x + self.dropout1(self.self_attn(h, k, v, attn_mask, positions, training=self.training))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN decoder block: self-attn (RoPE, causal) -> cross-attn (no positional term) -> FFN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.dropout,
            use_rope=(cfg.pos == "rope"),
            rope_base=cfg.rope_base,
        )
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, use_rope=False, rope_base=cfg.rope_base
        )
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor | None,
        cross_attn_mask: Tensor | None,
        positions: Tensor,
        memory_k: Tensor,
        memory_v: Tensor,
        self_cache: dict[str, Tensor] | None,
    ) -> Tensor:
        h = self.norm1(x)
        new_k, new_v = self.self_attn.project_kv(h, positions)
        if self_cache is not None:
            # Incremental decoding: concatenate this step's (already-rotated) K/V onto the cache
            # and grow it in place, so decode_step never re-rotates already-cached positions.
            if self_cache.get("k") is not None:
                new_k = torch.cat([self_cache["k"], new_k], dim=2)
                new_v = torch.cat([self_cache["v"], new_v], dim=2)
            self_cache["k"], self_cache["v"] = new_k, new_v
        k, v = new_k, new_v
        attn_out = self.self_attn(h, k, v, self_attn_mask, positions, training=self.training)
        x = x + self.dropout1(attn_out)

        h2 = self.norm2(x)
        cross_out = self.cross_attn(
            h2, memory_k, memory_v, cross_attn_mask, None, training=self.training
        )
        x = x + self.dropout2(cross_out)

        x = x + self.dropout3(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.enc_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor, attn_mask: Tensor | None, positions: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, positions)
        return self.final_norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor | None,
        cross_attn_mask: Tensor | None,
        positions: Tensor,
        memory_kv: list[tuple[Tensor, Tensor]],
        self_caches: list[dict[str, Tensor]] | None,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            cache = self_caches[i] if self_caches is not None else None
            memory_k, memory_v = memory_kv[i]
            x = layer(x, self_attn_mask, cross_attn_mask, positions, memory_k, memory_v, cache)
        return self.final_norm(x)


def _padding_key_mask(ids: Tensor, pad_id: int) -> Tensor:
    """(B,T) token ids -> (B,1,1,T) bool mask, True = attend (non-pad key), broadcastable over
    heads and query positions for F.scaled_dot_product_attention's boolean-mask convention.
    """
    return (ids != pad_id)[:, None, None, :]


class Transformer(nn.Module):
    """Deep-encoder / shallow-decoder translation model. Three-way tied embeddings
    (source embedding == target embedding == output projection weight), scaled by sqrt(d_model)
    at input time (Vaswani et al. 2017 §3.4).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        if cfg.pos == "sinusoidal":
            table = _sinusoidal_table(cfg.max_len, cfg.d_model)
            self.register_buffer("pos_table", table, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.cfg.d_model**-0.5)
        with torch.no_grad():
            self.embed.weight[self.cfg.pad_id].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def param_count(self) -> int:
        """Exact trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def _pos_add(self, x: Tensor, positions: Tensor) -> Tensor:
        if self.cfg.pos == "sinusoidal":
            # Decoding uses max_len = floor(1.5*src_len) + 10, which on a long source
            # can exceed cfg.max_len (the training-time table size) and index past the end of a
            # fixed-size buffer. Grow the table on demand rather than crash: re-registering a
            # larger buffer is a few KB and happens at most once per decode call that needs it.
            needed = int(positions.max().item()) + 1 if positions.numel() else 0
            if needed > self.pos_table.size(0):
                self.pos_table = _sinusoidal_table(needed, self.cfg.d_model).to(
                    device=self.pos_table.device, dtype=self.pos_table.dtype
                )
            return x + self.pos_table[positions].unsqueeze(0)
        return x  # RoPE is applied inside attention, not added to the embedding

    def encode(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """src: (B,T) token ids (EOS already appended by the loader, no BOS on the source side).
        src_key_padding_mask: (B,T) bool, True = real (non-pad) token. Returns (B,T,D) memory.
        """
        positions = torch.arange(src.size(1), device=src.device)
        x = self.embed_dropout(self._pos_add(self.embed(src) * self.embed_scale, positions))
        attn_mask = src_key_padding_mask[:, None, None, :]
        return self.encoder(x, attn_mask, positions)

    def decode(
        self,
        tgt_in: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        """Teacher-forced decode over the full target_in sequence (training path).
        tgt_in: (B,T) = BOS + target tokens. Returns logits (B,T,vocab_size).
        """
        t = tgt_in.size(1)
        positions = torch.arange(t, device=tgt_in.device)
        x = self.embed_dropout(self._pos_add(self.embed(tgt_in) * self.embed_scale, positions))
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device))
        self_mask = causal[None, None, :, :] & tgt_key_padding_mask[:, None, None, :]
        cross_mask = src_key_padding_mask[:, None, None, :]
        memory_kv = [layer.cross_attn.project_kv(memory, None) for layer in self.decoder.layers]
        out = self.decoder(x, self_mask, cross_mask, positions, memory_kv, self_caches=None)
        return F.linear(out, self.embed.weight)  # tied output projection, no bias (Vaswani et al.)

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        src_mask = src != self.cfg.pad_id
        tgt_mask = tgt_in != self.cfg.pad_id
        memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, tgt_mask, src_mask)

    def init_decode_cache(self, memory: Tensor, src_key_padding_mask: Tensor) -> dict:
        """Precompute cross-attention K/V once per decoder layer from encoder memory (they never
        change across decode steps), and set up empty per-layer self-attention KV caches.
        """
        cross_mask = src_key_padding_mask[:, None, None, :]
        memory_kv = [layer.cross_attn.project_kv(memory, None) for layer in self.decoder.layers]
        self_caches: list[dict[str, Tensor]] = [{} for _ in self.decoder.layers]
        return {
            "memory_kv": memory_kv,
            "self_caches": self_caches,
            "cross_mask": cross_mask,
            "step": 0,
        }

    def decode_step(self, tgt_token: Tensor, cache: dict) -> Tensor:
        """One incremental decode step. tgt_token: (B,1) token ids for the current step.
        Mutates `cache` (grows the self-attention KV cache, advances cache["step"]).
        Returns logits (B,1,vocab_size). RoPE positions use the absolute step offset so a token
        decoded at step 7 gets the same rotation whether reached incrementally or via `decode`.
        """
        step = cache["step"]
        positions = torch.arange(step, step + 1, device=tgt_token.device)
        x = self.embed_dropout(self._pos_add(self.embed(tgt_token) * self.embed_scale, positions))
        # No explicit self-attention mask: only real, already-generated positions are ever
        # cached, so every cached key is valid and causality is enforced structurally (a step
        # can only see keys cached at or before it).
        out = self.decoder(
            x, None, cache["cross_mask"], positions, cache["memory_kv"], cache["self_caches"]
        )
        cache["step"] = step + 1
        return F.linear(out, self.embed.weight)
