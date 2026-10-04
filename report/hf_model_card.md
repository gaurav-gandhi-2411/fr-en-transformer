---
license: other
license_name: research-evaluation-only
license_link: LICENSE
language:
- fr
- en
library_name: pytorch
pipeline_tag: translation
tags:
- translation
- from-scratch
- transformer
- pytorch_model_hub_mixin
datasets:
- Helsinki-NLP/opus-100
---

# fr-en-transformer

A French-to-English sentence translator, a 50M-parameter encoder-decoder transformer trained from scratch on OPUS-100 en-fr. No pretrained translation model, language model or tokenizer was used anywhere. This is model "v1" (training run `main`, final checkpoint, not averaged), decoded with beam 5, length penalty alpha 1.2 and segmentation of long inputs above 192 subword tokens. Ensembles, MBR decoding and longer-training extension runs were implemented but not evaluated for this release; v1 is the shipped model.

## Quick start

```bash
pip install git+https://github.com/gaurav-gandhi-2411/fr-en-transformer
```

```python
from nmt.translate import Translator

tr = Translator.from_pretrained("gauravgandhi2411/fr-en-transformer")
print(tr.translate(["La bibliothèque ferme plus tôt le dimanche."])[0])
```

This is not a `transformers` model; it needs the code in the GitHub repository above. The defaults of `translate` (beam 5, length penalty alpha 1.2, segmentation above 192 subword tokens) are the shipped configuration.

## Model

| | |
|---|---|
| Architecture | Encoder-decoder transformer, 8 encoder and 4 decoder layers, d_model 512, 8 heads, FFN 2048 (GELU), pre-LayerNorm with a final LayerNorm, dropout 0.1 |
| Positions | RoPE on encoder and decoder self-attention, none on cross-attention |
| Embeddings | Source, target and output projection share one 16,000 x 512 matrix (tied), scaled by sqrt(512) |
| Parameters | 50,229,248 |
| Attention | Hand-written, using `F.scaled_dot_product_attention`, with a decoder key-value cache. No `nn.Transformer` |
| Weights | `model.safetensors`, fp32, 200,944,776 bytes, sha256 `96ef3bb589b47c5c45ad6cf60a7b1c96ffca43ce3ea0377a28440648bf97757f` |

## Training data

OPUS-100 en-fr (`Helsinki-NLP/opus-100`), train split only: 1,000,000 pairs. After filtering, 921,670 pairs were used, 92.2% of the split. The OPUS-100 test split was not used.

| Step | Removed | Remaining |
|---|---|---|
| Raw train split | | 1,000,000 |
| Exact duplicate pairs | 37,122 | 962,878 |
| French identical to English | 13,061 | 949,817 |
| Character length ratio outside [1/3, 3] | 13,116 | 936,701 |
| More than 50% non-letter characters on a side | 10,412 | 926,289 |
| Leakage guard against evaluation sets (44 exact, 2,496 near-duplicate) | 2,540 | 923,749 |
| Hold-out for the E2 evaluation set, plus one near-duplicate of it | 1,001 | 922,748 |
| Pairs over 256 subword tokens on either side | 1,078 | 921,670 |

Text is normalised with NFKC, curly quotes unified and whitespace collapsed; casing is kept. The training set was not otherwise cleaned, so it inherits the noise of OPUS-100.

## Tokenizer

Joint French and English SentencePiece BPE, 16,000 pieces, byte fallback on, trained on the filtered training text only. `spm.model` sha256 starts `1fc208b5` (full: `1fc208b5b0885b8a1164ea2b6b44303d7acbddd709fab1f6b2dda53231779a37`).

## Training

- One NVIDIA L4, bf16 autocast, seed 1234, 24,645 optimizer steps (about 8.9 epochs), 11,791.8 s of training time.
- At least 25,000 padded tokens per optimizer step. AdamW (betas 0.9/0.98, weight decay 0.01), gradient clip 1.0.
- Learning rate: 4,000 warmup steps, peak 7e-4, linear decay to zero over the last 20% of steps.
- Label smoothing 0.1. Augmentation: with probability 0.15, 2 to 4 sentence pairs are concatenated (up to 256 tokens).

## Evaluation

BLEU and chrF come from the challenge's scoring script, which is not redistributed (BLEU on lowercased word and punctuation tokens, sentence-averaged chrF). Each cell is the point estimate with a 95% bootstrap interval (1,000 resamples, seed 1234). "Copy source" is a floor: the output is simply the French input, scored the same way. chrF gives partial credit for words shared with the reference, so a chrF in the 20s is not evidence of translation.

| Slice | n | v1 BLEU | v1 chrF | Copy-source BLEU | Copy-source chrF |
|---|---|---|---|---|---|
| Dev seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | 6.41 [3.49, 9.47] | 25.05 [20.37, 30.86] |
| Dev long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | 9.89 [3.20, 17.38] | 35.87 [30.40, 42.38] |
| Dev unseen domain | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | 1.76 [0.45, 3.64] | 17.93 [15.77, 20.36] |
| Dev pooled | 150 | 32.77 [26.98, 38.05] | 50.57 [46.86, 54.39] | 6.91 [3.53, 10.90] | 24.37 [21.84, 27.14] |
| Dev OVERALL | 150 | 42.29 [38.72, 45.82] | n/a | 16.10 [13.96, 18.74] | n/a |
| E1 | 1,940 | 35.98 [34.67, 37.30] | 54.98 [53.93, 56.11] | 6.03 [5.10, 7.12] | 26.56 [25.83, 27.33] |
| E2 | 1,000 | 38.02 [36.64, 39.39] | 61.78 [60.82, 62.71] | 5.19 [4.34, 6.12] | 31.98 [31.52, 32.52] |
| E2-synth | 300 | 37.73 [36.31, 39.07] | 63.31 [62.31, 64.29] | 5.42 [4.46, 6.46] | 34.79 [34.22, 35.44] |
| E3 | 1,000 | 19.45 [18.42, 20.52] | 42.31 [41.29, 43.27] | 1.25 [0.95, 1.54] | 20.47 [20.00, 20.98] |

Dev OVERALL is the task's official score: 0.4 BLEU + 0.4 chrF over all 150 dev sentences, plus 0.2 chrF on the unseen-domain slice. The copy-source floor is the only baseline here; same-size ablation runs at 1/6 of the training steps are in the report (`report/report.md` in the GitHub repository).

What each slice is, and which ones are proxies:

- **Dev** (150 sentences, provided with the task): a proxy for the hidden test set, split into seen (60), long (30) and unseen domain (60). Small, so the intervals are wide.
- **E1** (1,940): OPUS-100 validation sentences, minus those that overlap dev or test. In-domain proxy; also used to pick the checkpoint and decoding.
- **E2** (1,000): sentences held out of the training pool, French longer than 200 characters. Drawn from the training distribution and used for selection, so it is optimistic.
- **E2-synth** (300): concatenations of 2 to 4 E2 pairs (about 400 to 900 French characters). Synthetic; it measures handling of long inputs, not new content.
- **E3** (1,000): a random sample of OPUS Books en-fr. Never used for training or selection; the closest thing here to an out-of-domain test.

## Latency

CPU only, single sentences, v1 weights in fp32, beam 5, alpha 1.2, segmentation threshold 192. Hardware: AMD Ryzen 7 6800H (8 cores, 16 logical), 31.2 GiB RAM, torch 2.14.0+cpu, Python 3.13.5, Windows 11, Balanced power plan on AC power; one laptop shared with other jobs. Latency is the median over 4 runs of 200 E1 sentences (min to max across runs in brackets), including normalisation, tokenisation, decoding and detokenisation.

| Setting | p50 (ms) | p95 (ms) |
|---|---|---|
| Beam 5, torch default 8 threads | 287.8 (274.8 to 359.3) | 1,171.3 |
| Beam 5, 1 thread | 481.4 (416.9 to 522.5) | 1,950.6 |
| Greedy, default threads | 196.5 (194.7 to 202.3) | 695.6 |

Batches of 32 (320 E1 sentences, beam 5, default threads): 6.82 sentences/s (5.39 to 7.25). Peak resident memory is 596 MiB for single-sentence workers and 1,460 MiB (median) for batched workers. Several cells ran with background load, so treat these as indicative; absolute numbers do not transfer to a server CPU.

## Limitations

- Evaluation slices are small (dev has 30 to 60 sentences per slice), so intervals are wide and slice differences are often within noise.
- E2 comes from the training pool's distribution, and checkpoint and decoding choices were tuned on E1 and E2. Dev and E3 are the less biased numbers. Alpha 1.2 is the upper edge of the grid that was searched.
- Quality drops outside the training domain: chrF is 54.98 on E1 and 42.31 on E3 (books).
- One reference per sentence, and some OPUS-100 references are loose or unrelated to their source, so sentence scores can understate or overstate quality.
- Proper names, rare words and numbers can be mangled or left in French. Very long inputs (hundreds of words) are only partly covered; above 192 subword tokens the input is split on sentence punctuation and translated piece by piece.
- Trained only on OPUS-100, with no pretrained components. It was not evaluated on OPUS Books during training, and not on any other corpus, domain or language pair.
- Sentence-level only: no document context, no terminology control, no safety filtering.

## Intended use

Reading a French sentence or short paragraph in English, as a research demonstration of a from-scratch translator and as a baseline for small-compute experiments. Not for legal, medical, safety-critical or contractual translation, for any use where a wrong name, number or negation matters without human review, or for literary text. French to English only. No claim of parity with large pretrained systems is made; none was tested.

## Licence

These weights are released for research and evaluation use only. They were trained on OPUS-100 (`Helsinki-NLP/opus-100`, en-fr), whose card lists the licence as unknown; OPUS corpora carry mixed per-source licences, so the terms of the upstream corpora apply to any use of the weights. No warranty. The code in the GitHub repository is Apache-2.0 and is licensed separately.

## Provenance

Every number above comes from a file in the GitHub repository:

- Metrics and intervals, v1: `reports/final/main/seg_tuned/eval.json`
- Metrics and intervals, copy-source floor: `reports/final/baseline_copy_source/eval.json`
- Data filter counts, leakage guard, 922,748 pairs: `data/data_manifest.json`
- 921,670 pairs, epochs, tokens per step: `reports/epoch_accounting.json`
- Parameter count, precision, hardware, seed: `reports/main_l4/run_meta.json`
- Training time: `reports/main_l4/run_audit.json`
- Latency and memory: `reports/final/production/results.json` (tables in `reports/final/production/README.md`)
- Training configuration: `configs/main.yaml`
- Tokenizer settings: `nmt/data/tokenize.py`

Built with AI coding assistance; design, experiments and analysis are the author's.
