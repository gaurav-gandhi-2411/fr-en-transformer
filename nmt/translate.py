from __future__ import annotations

"""Public production API and CLI: `Translator.from_pretrained(repo_id).translate(...)`.
Applies the same normalization as training, calls decode.py, and reports batched
inference latency/throughput. Spec §7.
"""
