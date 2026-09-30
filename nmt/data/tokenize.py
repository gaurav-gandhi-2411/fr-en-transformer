from __future__ import annotations

"""SentencePiece BPE tokenizer training (joint 16k vocab, byte fallback) and
pre-tokenized uint16 numpy shard generation plus a manifest with sha256 hashes,
counts and length histograms. Pushes shards to the private HF dataset repo. Spec §4.
"""
