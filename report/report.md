# French-to-English Transformer from Scratch: Design, Results and Generalization

Gaurav Gandhi · 4 October 2026

**Summary.** The model scores 42.29 [38.72, 45.82] OVERALL on dev, against 16.10 for copying the source. It is strong on long inputs (dev long chrF 63.16), and its open problem is literature, where it sits 12.67 chrF below in-domain text.

I built a 50.2M-parameter encoder-decoder from scratch on OPUS-100 en-fr, with no pretrained model, tokenizer or language-ID tool. I pre-registered my design and selection decisions, selected checkpoints and decoding settings on held-out proxies only, and ran an automated verification pass that re-derived every reported number; apart from my own hours, each one comes from a file in the repository (mostly under `reports/`). The submitted model ("v1") is the final checkpoint of the main run, decoded with beam 5, a GNMT length penalty of 1.2, a 3-gram repeat block, and sentence splitting above 192 tokens.

## 1. Architecture and Design Decisions

| Decision | Alternative considered | Rationale and evidence |
|---|---|---|
| Encoder-decoder [1] | Decoder-only (prefix LM) | Translation is conditional: a bidirectional encoder plus explicit cross-attention is the natural fit at 0.9M pairs. This is a design prior; a decoder-only comparison is in my next steps (Section 5). |
| 8 encoder / 4 decoder layers, d=512, 8 heads, FFN 2048, pre-LN [3], GELU, tied embeddings: 50,229,248 parameters | Transformer-big; symmetric 6/6 | The decoder runs once per output token, so I kept it shallow to cut decoding cost [4]; a 6/6 comparison comes next. I sized the model to the budget: 24,645 steps (8.86 epochs), 3.3 h on one L4. |
| Attention written by hand on PyTorch's `scaled_dot_product_attention`, with a KV cache | `torch.nn.Transformer` | I needed RoPE on queries and keys and an incremental cache for beam search; `nn.Transformer` supports neither. |
| RoPE [2] | Sinusoidal; ALiBi [9] | Measured gain: +0.89 chrF [+0.66, +1.14] on long sentences (E2) and +3.11 [+2.40, +3.85] on very long synthetic inputs. The ablation ran 1.8 epochs, so I treat transfer to the full run as an assumption. |
| Concatenation augmentation [6], p=0.15 (2 to 4 pairs) | None | It did not meet its pre-registered criterion (Section 4); the main run was trained before that result. |
| Joint SentencePiece BPE, 16k, byte fallback | Separate vocabularies, Unigram, BPE-dropout | A small shared vocabulary suits 0.9M pairs [7], copies names across languages and allows three-way tying. Zero unknown tokens on dev, test, E1, E2 and E3. A vocabulary sweep comes next. |
| WSD schedule: warmup 4,000, peak 7e-4, linear decay over the last 20% [5] | Cosine, inverse-sqrt | Any checkpoint can start its own cooldown, which suits preemptible Colab. The final checkpoint beat both averaged candidates (48.75 against 48.66 and 48.69 on the selection objective). |
| bf16, label smoothing 0.1, dropout 0.1 | fp16 with loss scaling | Native on the L4, and no optimizer step was skipped in the logged training history. Regularisation is at standard values. |
| Beam 5, alpha 1.2, 3-gram block, split above 192 tokens, output never empty (beam, then greedy, then a copy of the source) | Greedy; splitting at 64 tokens | Alpha 1.2 was the top of my grid in all 6 tuning runs, so extending the grid is an easy next win. Splitting moved the main model by at most +0.07 chrF. |
| Selection on 0.4 BLEU + 0.4 chrF (E1+E2) + 0.2 chrF(E1) | Selecting on dev or books | The selection code can only load E1 and E2; v1 scores 48.7455. |

## 2. Engineering Challenges and Resolutions

- **Recovering the GPU budget.** Before the main run, I replayed an epoch of the batcher and found that joining sentence pairs after length-grouping left 80.6% of every batch as padding, with micro-batches up to 131,584 tokens against a budget of 8,192. Planning the joins first cut padding to 29.6% and capped batches at 8,192, before any main-run training step.
- **A scorer I could trust on any OS.** On Windows the official scorer read UTF-8 as cp1252, so the dev references scored against themselves reached only BLEU 97.50. I kept the scorer byte-for-byte unchanged and run it in UTF-8 mode; the same file scores 100, and a test guards it.
- **Memory-safe resumes.** Loading checkpoints put a second copy of the weights and optimizer state on an 8 GB GPU, ending all 8 resumes of one run at step 6. Checkpoints now load to CPU first, with a regression test.
- **A portable pipeline.** When Colab moved to Python 3.13, I pinned Colab's own CUDA torch through a constraints file and added 3.13 to CI.
- **Working within compute limits.** When a shared GPU stalled an ablation for 5.7 h, I moved all ablations to one Colab L4 session, so they ran under identical conditions.
- **Reproducibility by default.** Pinned dependencies, seed 1234 everywhere, Colab runs from git tags, and a test that a resumed run matches an uninterrupted one.

## 3. Results by Slice

Official scorer, 95% bootstrap intervals from 1,000 resamples [8]. "Copy" is a floor: the French source submitted as the translation. E1, E2 and E3 are larger proxies I built from public data (Section 6); E2-synth is 300 inputs of 400 to 900 French characters made by joining 2 to 4 E2 pairs.

| Set | n | v1 BLEU | v1 chrF | Copy BLEU | Copy chrF |
|---|---|---|---|---|---|
| Dev seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | 6.41 [3.49, 9.47] | 25.05 [20.37, 30.86] |
| Dev long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | 9.89 [3.20, 17.38] | 35.87 [30.40, 42.38] |
| Dev unseen | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | 1.76 [0.45, 3.64] | 17.93 [15.77, 20.36] |
| Dev all | 150 | 32.77 [26.98, 38.05] | 50.57 [46.86, 54.39] | 6.91 [3.53, 10.90] | 24.37 [21.84, 27.14] |
| Dev OVERALL | 150 | 42.29 [38.72, 45.82] | | 16.10 [13.96, 18.74] | |
| E1 | 1,940 | 35.98 [34.67, 37.30] | 54.98 [53.93, 56.11] | 6.03 [5.10, 7.12] | 26.56 [25.83, 27.33] |
| E2 | 1,000 | 38.02 [36.64, 39.39] | 61.78 [60.82, 62.71] | 5.19 [4.34, 6.12] | 31.98 [31.52, 32.52] |
| E2-synth | 300 | 37.73 [36.31, 39.07] | 63.31 [62.31, 64.29] | 5.42 [4.46, 6.46] | 34.79 [34.22, 35.44] |
| E3 | 1,000 | 19.45 [18.42, 20.52] | 42.31 [41.29, 43.27] | 1.25 [0.95, 1.54] | 20.47 [20.00, 20.98] |

The dev slices hold only 30 to 60 sentences, so I base conclusions on the larger sets. E2 shares the training distribution and E2-synth is built from E2, so both measure length handling rather than new content. E3 chrF (42.31) falls inside the dev-unseen interval [40.74, 48.39], which supports using it as a proxy. COMET scoring is a next step.

**Failure analysis.** The failure modes are narrow and well understood. Long inputs are a strength (chrF 60.89 at 41 to 80 words, 60.32 above 80); very short inputs are the hardest (48.16 at 10 words or fewer), likely because sentence-level chrF penalises one wrong word more in a short sentence. The rarest fifth of sentences scores 50.45, against 55.19 and 56.05 for the two most common fifths. Output errors are rare on E1: 2.2% truncated (under half the reference length), 4.0% overlong (over 1.5x), and 0.5% with a repeated 3-gram. Repetition on E2 (3.0%) sits well below the references' own 14.7%, and a strict untranslated-copy check found none.

## 4. Generalization and the Domain Gap

On dev, moving from seen to unseen-domain text costs 5.32 chrF (50.08 to 44.76) and 10.43 BLEU (32.43 to 21.99), with overlapping intervals. On the larger proxies the gap is unambiguous: E1 to E3 is -12.67 chrF and -16.52 BLEU. Length is not the driver: dev long (63.16) and E2 (61.78) both score above their seen counterparts.

- **H1, RoPE over sinusoidal: supported** (Section 1).
- **H2, concatenation for long inputs: not supported.** E2 chrF moved +0.06 [-0.19, +0.29] (p = 0.313), short of the pre-registered criterion, although it helped on synthetic long inputs (+1.09 [+0.59, +1.68]) and was non-inferior in-domain (lower bound -0.22).

**Where the domain gap comes from.** My pre-registered regression of sentence chrF on length, repetition, source rarity, dialogue punctuation and a domain flag (E1 and E3, n = 2,940, robust errors) puts the gap almost entirely on the domain flag (+12.88; the four covariates net -0.21). A richer, exploratory five-group feature model explains -7.5% to 40.5% of the gap depending on the decomposition (8.7% in the primary model), so the remaining 91.3% is unmeasured variation rather than a measured domain effect. My working hypothesis is literary vocabulary and style that OPUS-100 under-represents, with rare target-side words contributing +2.34 chrF [+1.80, +2.99]; testing it with a targeted intervention is the natural next experiment. Reference noise accounts for only about 0.22 chrF on E3.

## 5. Next Steps

First, I would run the final selection I have already implemented and pre-registered: the longer-trained branches (validation loss 2.904 for v1, then 2.875 and 2.870), ensembles, MBR decoding with a chrF utility, and an alpha grid beyond 1.2. Next, the comparisons I deferred to stay within budget: a decoder-only model, a symmetric 6/6 encoder-decoder, and a vocabulary-size sweep. After that: back-translation, BPE-dropout or R-Drop, a larger model, and several seeds per experiment.

## 6. Data and Constraints

OPUS-100 en-fr (revision 805090dc), train split: 1,000,000 pairs. I removed 37,122 duplicates, 13,061 pairs whose French equals the English, 13,116 with a length ratio outside [1/3, 3], 10,412 that were mostly non-letters, 2,540 matching evaluation sentences exactly or nearly, 1,001 held out for E2, and 1,078 longer than 256 tokens, leaving **921,670 pairs (92.2%)**. The leakage guard exists because OPUS-100 contains near-copies of evaluation sentences; after it, the overlap check finds zero. E1 is the OPUS-100 validation split (1,940 pairs), E2 is 1,000 long held-out pairs, and E3 is 1,000 opus_books pairs used only for reporting: never trained, tuned or selected on. One NVIDIA L4 (24 GB), bf16, seed 1234, no paid APIs. The 24,645-step schedule in `configs/main.yaml` came from a pilot run.

## 7. Effort, Compute and Links

About 8 to 9 hours of my hands-on time over about three days, plus 10.1 GPU-hours of training on one L4 (main 3.3 h, ablations 1.6 h, extension 5.0 h, pilot 0.3 h). I used AI coding assistance; the design, experiments and analysis are mine.

Code: <https://github.com/gaurav-gandhi-2411/fr-en-transformer> · Model: <https://huggingface.co/gauravgandhi2411/fr-en-transformer> · W&B: <https://wandb.ai/gauravgandhi429-gaurav-gandhi/fr-en-transformer-public>

Reproduce: `python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234` (one L4, about 3.3 h of training).

## References

[1] Vaswani et al. 2017. Attention Is All You Need. NeurIPS. arXiv:1706.03762. [2] Su et al. 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864. [3] Xiong et al. 2020. On Layer Normalization in the Transformer Architecture. ICML. arXiv:2002.04745. [4] Kasai et al. 2021. Deep Encoder, Shallow Decoder. ICLR. arXiv:2006.10369. [5] Hägele et al. 2024. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS. arXiv:2405.18392. [6] Nguyen, Murray, Chiang. 2021. Data Augmentation by Concatenation for Low-Resource Translation. IWSLT. [7] Sennrich, Zhang. 2019. Revisiting Low-Resource Neural Machine Translation. ACL. [8] Koehn. 2004. Statistical Significance Tests for Machine Translation Evaluation. EMNLP. [9] Press, Smith, Lewis. 2022. Train Short, Test Long (ALiBi). ICLR. arXiv:2108.12409.
