# fr-en-transformer

A French->English encoder-decoder transformer, trained from scratch (no pretrained
translation models or LMs) on `Helsinki-NLP/opus-100` (en-fr), within a free Colab T4
budget. Built for the take-home challenge: hand-written model (RoPE/sinusoidal switch,
deep-encoder/shallow-decoder), a from-scratch SentencePiece tokenizer, a leakage-guarded
data pipeline, and a statistically honest sliced evaluation with bootstrap CIs.

**Status: work in progress (P0).**

Safety-net test predictions (v1, run `main`, candidate `final`): see `submission/README.md`.

## Reproduce

```
python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234
```

## Data provenance

The provided dev/test files and the official scorer are vendored byte-identical; their
sha256 hashes are recorded in [`official/SHA256SUMS`](official/SHA256SUMS) and checked
in `tests/test_official_scorer.py`.

## Windows note: scoring needs UTF-8 mode

`official/score.py` (vendored byte-identical, so it cannot be patched) opens its inputs with the
platform default encoding, which is cp1252 on Windows. A UTF-8 prediction file with literal
accented text is then decoded as mojibake: a reference-identical dev prediction file scored
BLEU 97.5 instead of 100. Run the scorer through `nmt.evaluate.run_official_scorer` (used by
`scripts/eval_local.py` and the tests), which sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`, or
set `PYTHONUTF8=1` yourself before calling `python official/score.py` by hand.
