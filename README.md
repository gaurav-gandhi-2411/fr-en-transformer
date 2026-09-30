# fr-en-transformer

A French->English encoder-decoder transformer, trained from scratch (no pretrained
translation models or LMs) on `Helsinki-NLP/opus-100` (en-fr), within a free Colab T4
budget. Built for the take-home challenge: hand-written model (RoPE/sinusoidal switch,
deep-encoder/shallow-decoder), a from-scratch SentencePiece tokenizer, a leakage-guarded
data pipeline, and a statistically honest sliced evaluation with bootstrap CIs.

**Status: work in progress (P0).**

## Reproduce

```
python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234
```

## Data provenance

The provided dev/test files and the official scorer are vendored byte-identical; their
sha256 hashes are recorded in [`official/SHA256SUMS`](official/SHA256SUMS) and checked
in `tests/test_official_scorer.py`.
