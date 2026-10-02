# French-to-English Transformer from scratch: decisions, results, gap analysis

DRAFT, private. Double-brace items are filled later from the files in `report/PLACEHOLDERS.md`. "v1" is the safety-net submission (`main`, checkpoint `final`, tuned decoding).

I trained a 50.2M-parameter encoder-decoder on 921,670 OPUS-100 en-fr pairs, with no pretrained model of any kind. Each number comes from a file in `reports/`.

## 1. Architecture decisions

| Choice | Rejected | Trade-off | Evidence |
|---|---|---|---|
| 8 encoder, 4 decoder layers, d=512, 8 heads, FFN 2048, pre-LN, tied embeddings: 50,229,248 parameters | Transformer-big, decoder-only | Cheaper decoding than a symmetric stack. I trained no symmetric baseline, so the speed benefit is the paper's, not mine | `reports/main_l4/run_meta.json` `config.param_count`. E1 validation loss (20 micro-batches, label-smoothed) was at its minimum, 2.9041, at the last evaluation (`run_audit.json`) |
| RoPE | Sinusoidal; ALiBi, not tried | The ablation ran 4,107 steps (1.8 epochs), so it says little about 24k | H1 supported: chrF delta on E2 +0.89 [+0.66, +1.14], on E2-synth +3.11 [+2.40, +3.85] (`reports/final/compare/H1_*_seg_off.json`). With segmentation on, the E2-synth delta is +0.74 [+0.40, +1.05] |
| Concatenation augmentation, p=0.15, 2 to 4 pairs | None | Helps only inputs shaped like the augmentation | H2 not supported: E2 +0.06 [-0.19, +0.29], p=0.313; E2-synth +1.09 [+0.59, +1.68]; E1 non-inferiority met (lower bound -0.22) (`H2_*_seg_off.json`). `main` was trained with it before this result existed |
| Joint SentencePiece BPE, 16k, byte fallback | Unigram, BPE-dropout | Small vocabulary, longer sequences. I ran no vocabulary or model-type sweep | sha256 `1fc208b5...`; no UNK or byte fallback on dev, test, E1, E2, E3 (`reports/tokenizer_stats.json`) |
| WSD schedule: warmup 4,000, peak 7e-4, linear decay over the last 20% | Cosine, inverse-sqrt, not tried | Any stable checkpoint can start a cooldown, which the extension run uses | `wsd_lr_scale`, `nmt/train.py:564-590`. Averaging 5 or 4 checkpoints did not beat the final one: objective 48.66 and 48.69 against 48.75 (`reports/final/main/selection.json`; no CI) |
| Label smoothing 0.1, dropout 0.1 | Stronger regularisation | Not tuned; the trigger was more than 10 epochs and `main` ran 8.86 | PREREG 2026-10-02; `reports/epoch_accounting.json` |
| Beam 5, GNMT length penalty alpha 1.2, 3-gram block, segmentation above 192 source tokens | Greedy; T=64 | Alpha 1.2 was the top of the grid in all 6 tuning runs, so the optimum may be higher. Segmentation moved `main` by at most +0.07 chrF | `selection.json`; segmentation tables in `reports/final/SUMMARY.md` |
| Selection on 0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1) | Selecting on dev or E3 | E2 is held out of the training pool but in-domain, so it flatters the winner | `nmt/selection.py` can load only E1 and E2; objective 48.7455 (`selection.json`) |
| MBR with chrF utility; log-probability ensembles | Single model, beam only | 8 or 16 candidates per sentence plus a pairwise chrF pass, slower than beam 5 | {{FINAL_SELECTED_CONFIG}}; objective {{FINAL_OBJECTIVE}} against v1 48.7455 |
| bf16 autocast, no loss scaling | fp16 with GradScaler, the spec's T4 plan | Compute moved to an L4, which supports bf16 | `run_meta.json` `config.precision`; no skipped optimizer step in 24,645 (`run_audit.json`, `grad_skip_count` 0) |

## 2. References

1. Vaswani et al. 2017. Attention Is All You Need. NeurIPS 30. arXiv:1706.03762.
2. Su, Lu, Pan, Murtadha, Wen, Liu. 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
3. Xiong et al. 2020. On Layer Normalization in the Transformer Architecture. ICML 2020. arXiv:2002.04745.
4. Kasai, Pappas, Peng, Cross, Smith. 2021. Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation. ICLR 2021. arXiv:2006.10369.
5. Hägele et al. 2024. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024. arXiv:2405.18392.
6. Nguyen, Murray, Chiang. 2021. Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution. IWSLT 2021.
7. Sennrich, Zhang. 2019. Revisiting Low-Resource Neural Machine Translation: A Case Study. ACL 2019.
8. Koehn. 2004. Statistical Significance Tests for Machine Translation Evaluation. EMNLP 2004.

## 3. Challenges

**Concatenation padding.** The augmentation joined pairs after bucketing and padded the whole batch to the longest joined row. At micro-batch 4,096 and p=0.15, 80.6% of padded tokens were padding and micro-batches reached 131,584 tokens against a budget of 8,192 (PREREG 2026-10-02), before any S3 or main step ran. I now draw the concat plan before bucketing (commit `ecb1a85`); padding is 29.6% and the largest micro-batch is 8,192 (`reports/epoch_accounting.json`).

**Resume out-of-memory.** Commit `dc6e7e8`: `torch.load` put the checkpoint back on the GPU and `train()` kept it, a second copy of weights and AdamW state. On the 8 GB RTX 3070 the S1 attempt ran out of memory at step 6 on all 8 resumes; the 2.7M-parameter smoke model could not show it. Checkpoints now load to CPU, with a regression test.

**Python 3.13 on Colab.** The first tag, `v0.2-colab`, failed at install on Colab's new 3.13 before Drive was mounted. `v0.2.1-colab` supports 3.12 and 3.13, keeps Colab's own torch and adds a preflight cell; CI runs both (PLAN.md).

**Scorer encoding.** `official/score.py` opens files in the platform encoding, cp1252 on Windows, so a prediction file identical to the references scored BLEU 97.5, not 100 (README). The scorer is byte-pinned, so I wrapped it: `nmt.evaluate.run_official_scorer` sets `PYTHONUTF8=1`.

**GPU contention.** The 3070 was shared with another of my jobs. S1 waited 5.7 h, was stopped at step 3, then hit the error above. Ablations must share hardware, so I retired the 3070 and ran S1, S2 and S3 in one Colab L4 session (PLAN.md, 2026-10-02). I also corrected my own commit message `9e36831`, which blamed the wrong cause.

**Stacked PR merge.** PRs #25 and #26 merged into their stack bases, not main, minutes after #24 reached main, so main held only #24. PR #27 carried the same head onto main and merged at 18:38 UTC. No code changed. I now retarget every PR of a stack to main before merging any.

**COMET cost.** COMET-22 over 17,294 distinct triples was projected at 2.2 h on CPU, 4.5 h for all triples (estimates, `HANDOFF.md`). I stopped after 5 of 7 chunks and kept the output marked incomplete (`reports/final/comet_partial_cpu_INCOMPLETE/`). COMET: {{FINAL_COMET_SUMMARY}}.

**Compute and effort.** `main`: 11,791.8 s of training on an L4 (`reports/main_l4/run_audit.json`), about 5.5 CU at the 1.54 CU/h rate I reported (estimate). Ablations S1, S2, S3: {{ABL_WALL_S1}}, {{ABL_WALL_S2}}, {{ABL_WALL_S3}} s. Extension run: planned 38,500 steps at 0.4785 s, about 5.12 h (estimate), measured {{EXT_TRAIN_WALL_S}} s. Final selection on Colab: {{FINAL_COLAB_HOURS}} h. My own time: {{GG_EFFORT_HOURS}} h.

## 4. Dev results by slice

Official scorer, 1,000-resample bootstrap 95% CIs. v1 is `main`/`final`, alpha 1.2, beam 5, T=192 (`reports/final/main/seg_tuned/eval.json`). Submitted model: {{FINAL_MODEL_NAME}}.

| Set | n | v1 BLEU | v1 chrF | Final BLEU | Final chrF |
|---|---|---|---|---|---|
| Dev seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | {{FINAL_DEV_SEEN_BLEU}} | {{FINAL_DEV_SEEN_CHRF}} |
| Dev long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | {{FINAL_DEV_LONG_BLEU}} | {{FINAL_DEV_LONG_CHRF}} |
| Dev unseen | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | {{FINAL_DEV_UNSEEN_BLEU}} | {{FINAL_DEV_UNSEEN_CHRF}} |
| Dev OVERALL | 150 | 42.29 [38.72, 45.82] | | {{FINAL_DEV_OVERALL}} | |
| E1 | 1,940 | 35.98 [34.67, 37.30] | 54.98 [53.93, 56.11] | {{FINAL_E1_BLEU}} | {{FINAL_E1_CHRF}} |
| E2 | 1,000 | 38.02 [36.64, 39.39] | 61.78 [60.82, 62.71] | {{FINAL_E2_BLEU}} | {{FINAL_E2_CHRF}} |
| E2-synth | 300 | 37.73 [36.31, 39.07] | 63.31 [62.31, 64.29] | {{FINAL_E2SYNTH_BLEU}} | {{FINAL_E2SYNTH_CHRF}} |
| E3 | 1,000 | 19.45 [18.42, 20.52] | 42.31 [41.29, 43.27] | {{FINAL_E3_BLEU}} | {{FINAL_E3_CHRF}} |

The 60 and 30 sentence slices have wide intervals, so I draw conclusions from E1, E2 and E3. E2 comes from the training pool and E2-synth is built from E2 sentences, so neither tests unseen content. COMET-22: {{FINAL_COMET_SUMMARY}}. I have no trivial-baseline number; the only weaker reference is the S1 ablation at 4,107 steps (dev OVERALL 37.52, `reports/final/SUMMARY.md`).

## 5. Generalization gap

Specified in spec.md section 10 before any run: OLS of sentence chrF on length deviation, repetition, source rarity, dialogue punctuation and a domain flag, over E1 and E3 (n=2,940), HC3 errors. For `main` the E1-to-E3 chrF gap is 12.67. The domain flag carries +12.88 of it (101.6%); the four measured covariates net to -0.21 (length -0.49, repetition +0.05, rarity +0.08, dialogue +0.16); R-squared 0.178 (`reports/final/main/seg_tuned/analysis.json`, `gap_decomposition`). What I could measure explains none of the gap, so it is "something about the books" that I have not isolated. Reference noise accounts for about 0.22 chrF on E3 (`metric_artifact_share`). E3 chrF [41.29, 43.27] lies inside the dev unseen interval [40.74, 48.39], so E3 is a fair proxy.

Exploratory v2, labelled as such: {{GAPV2_SUMMARY}} Findings: {{GAPV2_FINDINGS}} Caveats: {{GAPV2_CAVEATS}}

## 6. What I would do next

Train a naive and a symmetric 6/6 baseline so the shallow-decoder claim has a number. Sweep vocabulary size and compare unigram with BPE. Add BPE-dropout and back-translation from OPUS-100 English text, never books. Tune alpha past 1.2 on a held-out set the tuning never sees. Distil the best system for CPU. Extension result: {{EXT_SUMMARY}}.

## 7. Links

Private repo: https://github.com/gaurav-gandhi-2411/fr-en-transformer. Public repo: {{LINK_REPO}}. Model: {{LINK_HF_MODEL}}. W&B: {{LINK_WANDB}}. Private HF repos, by name only: `OWNER/fr-en-transformer-data` (dataset) and `OWNER/fr-en-transformer-eval` (model).

Built with AI coding assistance; design, experiments and analysis are mine.
