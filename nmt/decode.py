from __future__ import annotations

"""Batched decoding: beam search (GNMT length penalty, n-gram repetition block),
greedy fallback, and the long-input segmentation fallback. Never emits an empty
string. Spec §7.
"""
