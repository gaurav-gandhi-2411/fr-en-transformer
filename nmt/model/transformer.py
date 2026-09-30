from __future__ import annotations

"""Hand-written encoder-decoder transformer: 8 encoder / 4 decoder layers, d=512,
8 heads, FFN 2048, pre-LayerNorm, three-way tied embeddings, RoPE/sinusoidal
positional-encoding switch, F.scaled_dot_product_attention, decoder KV cache for
incremental decoding. Spec §5.
"""
