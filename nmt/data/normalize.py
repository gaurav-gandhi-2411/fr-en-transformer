from __future__ import annotations

"""Shared text normalization used identically at train and inference time.

Will provide `normalize_text`: NFKC normalization, apostrophe/quote unification
(`' ' ʼ` -> `'`, `« » " "` -> `"`), whitespace collapse and strip, casing preserved.
Used by prepare.py, tokenize.py, translate.py and evaluate.py — never re-implemented
elsewhere (see PLAN.md decisions). Spec §3.
"""
