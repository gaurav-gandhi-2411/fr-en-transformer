# fr-en-transformer

A French-to-English encoder-decoder Transformer trained from scratch on OPUS-100 en-fr: hand-written model code (RoPE, SDPA attention, KV cache), a joint 16k SentencePiece BPE tokenizer, a leakage-guarded data pipeline, resumable single-GPU training, beam-search decoding with a length penalty, and a sliced evaluation with bootstrap confidence intervals. No pretrained model, tokenizer or language-ID tool is used anywhere. The 50.2M-parameter model trained for 3.3 h on one NVIDIA L4 (bf16, seed 1234) and scores 42.29 [38.72, 45.82] on the 150-sentence official dev OVERALL, against 16.10 for copying the French source. Decisions, the generalization analysis and the failures are in [`report/report.md`](report/report.md) (the PDF is attached to the release); the analysis plan fixed before the results is [`PREREG.md`](PREREG.md).

## Results

Official scorer, 1,000-resample bootstrap 95% CIs, model "v1" (run `main`, checkpoint `final`, beam 5, alpha 1.2, segmentation above 192 source tokens). Source: [`reports/final/main/seg_tuned/eval.json`](reports/final/main/seg_tuned/eval.json) and [`reports/final/baseline_copy_source/eval.json`](reports/final/baseline_copy_source/eval.json).

| Set | n | BLEU | chrF | Copy-source BLEU | Copy-source chrF |
|---|---|---|---|---|---|
| Dev seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | 6.41 | 25.05 |
| Dev long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | 9.89 | 35.87 |
| Dev unseen domain | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | 1.76 | 17.93 |
| Dev all | 150 | 32.77 [26.98, 38.05] | 50.57 [46.86, 54.39] | 6.91 | 24.37 |
| E1 (seen proxy) | 1,940 | 35.98 [34.67, 37.30] | 54.98 [53.93, 56.11] | 6.03 | 26.56 |
| E2 (long proxy) | 1,000 | 38.02 [36.64, 39.39] | 61.78 [60.82, 62.71] | 5.19 | 31.98 |
| E2-synth (synthetic long) | 300 | 37.73 [36.31, 39.07] | 63.31 [62.31, 64.29] | 5.42 | 34.79 |
| E3 (books proxy) | 1,000 | 19.45 [18.42, 20.52] | 42.31 [41.29, 43.27] | 1.25 | 20.47 |

The dev slices have 30 to 60 sentences and wide intervals; E1, E2 and E3 are built from public data. E2 comes from the training pool and E2-synth is built from E2, so neither measures unseen content. E3 (`opus_books`) is the unseen-domain proxy and was never trained on or used for selection.

## Quickstart

Install (Python 3.12 or 3.13):

```bash
uv sync --frozen          # or: pip install -r requirements.txt
```

The data pipeline needs the evaluation package files that were provided for the challenge; they are not in this repository. Place them at `data/dev/inputs.jsonl`, `data/dev/labels.jsonl`, `data/test/inputs.jsonl`, `data/test/sample_submission.json` and `official/score.py`, then check them from the repository root with `sha256sum -c official/SHA256SUMS` (the hashes are published). Without them the pipeline stops at the `prepare` stage and the tests that depend on them are skipped.

The one seeded reproduce command:

```bash
python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234
```

This downloads OPUS-100 and `opus_books`, prepares the data, trains the tokenizer and model, tunes decoding on E1 and E2, evaluates and analyses. The published model was trained on one L4 with the same schedule (24,645 steps, 8.86 epochs, about 3.3 h) through [`colab/train.ipynb`](colab/train.ipynb), which also streams checkpoints to Drive. A CPU smoke run of the same command: `python -m nmt.pipeline --config configs/smoke.yaml --stage all --seed 1234`.

On Windows set `PYTHONUTF8=1`: the official scorer reads files in the platform encoding (cp1252), which turns accented UTF-8 into mojibake. `nmt.evaluate.run_official_scorer` sets it for you.

## Translate with the released model

```python
# pip install git+https://github.com/gaurav-gandhi-2411/fr-en-transformer
from nmt.translate import Translator

tr = Translator.from_pretrained("gauravgandhi2411/fr-en-transformer")
print(tr.translate(["Le chat dort sur le canapé."]))  # defaults: the shipped config
```

## Repository layout

| Path | Contents |
|---|---|
| `nmt/model/` | Encoder-decoder Transformer, RoPE or sinusoidal positions, KV cache |
| `nmt/data/` | Normalisation, filtering, leakage guard, eval-set construction, tokenizer, token-bucketed loader |
| `nmt/train.py`, `nmt/decode.py`, `nmt/translate.py` | Resumable WSD training, beam search, batched translation API |
| `nmt/evaluate.py`, `nmt/compare.py`, `nmt/analysis.py` | Official-scorer evaluation, bootstrap CIs and paired tests, error and gap analysis |
| `nmt/selection.py`, `nmt/tune.py` | Selection objective and decoding search on E1 and E2 |
| `nmt/mbr.py`, `nmt/ensemble.py`, `nmt/final_all.py` | MBR, ensembling and staged final selection: implemented and pre-registered, not evaluated |
| `configs/` | Run configs: `main`, `smoke`, ablations S1 to S3, extension runs |
| `colab/` | Colab notebook and helpers used for the L4 runs |
| `reports/` | Result files every reported number comes from (`reports/final/`) |
| `report/` | The 3-page report and the Hugging Face model card |
| `scripts/` | Report, audit and analysis scripts, and `local_ci.py` |
| `tests/` | Unit and integration tests |

## Evaluation and selection protocol

Checkpoints and decoding settings (beam, alpha, segmentation threshold) are chosen only on E1 (OPUS-100 validation split, 1,940 pairs after leakage removal) and E2 (1,000 long pairs held out of training) with a fixed objective: 0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1). Dev, E2-synth and E3 are reported once for the selected model and never used to choose anything. Hypotheses H1 (RoPE over sinusoidal) and H2 (concatenation augmentation) were pre-registered with decision rules in [`PREREG.md`](PREREG.md). Training never sees `opus_books`; train pairs that match any dev or test source, dev reference, E1 or E3 sentence are removed, and a post-check finds 0 hits.

## Links

Report: [`report/report.md`](report/report.md) (PDF in the release). Model: <https://huggingface.co/gauravgandhi2411/fr-en-transformer>. W&B: see the report. Pre-registration: [`PREREG.md`](PREREG.md).

## Licence and data

Code: Apache-2.0 ([`LICENSE`](LICENSE)). Model weights: licence "other", research and evaluation use only; they were trained on OPUS-100 (`Helsinki-NLP/opus-100`), whose card states no licence, so the terms of the upstream corpora apply. See the model card.

Built with AI coding assistance; design, experiments and analysis are mine.
