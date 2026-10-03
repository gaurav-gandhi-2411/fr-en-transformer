# French-to-English Transformer from scratch: decisions, results, gap analysis

DRAFT, private; double-brace items are filled from `report/PLACEHOLDERS.md`. "v1" is the safety-net submission (`main`, checkpoint `final`, tuned decoding).

I trained a 50.2M-parameter encoder-decoder on 921,670 OPUS-100 en-fr pairs, with no pretrained model of any kind. Each number comes from a file in the repository, named where used.

## 1. Architecture decisions

| Choice | Rejected | Trade-off | Evidence |
|--------------|----------|-------------|--------------------------|
| 8 encoder, 4 decoder layers, d=512, 8 heads, FFN 2048, pre-LN, tied embeddings: 50,229,248 parameters | Transformer-big, decoder-only | Cheaper decoding than a symmetric stack; I trained no symmetric baseline, so the speed benefit is the paper's | `reports/main_l4/run_meta.json` `config.param_count`. E1 validation loss (20 micro-batches, label-smoothed) was at its minimum, 2.9041, at the last evaluation (`run_audit.json`) |
| RoPE | Sinusoidal (my ablation); ALiBi [9], not run by me: rejected on prior reading, not on my own ablation | The ablation ran 4,107 steps (1.8 epochs), so it says little about 24k | H1 supported: chrF delta on E2 +0.89 [+0.66, +1.14], on E2-synth +3.11 [+2.40, +3.85] (`reports/final/compare/H1_*_seg_off.json`). With segmentation on, the E2-synth delta is +0.74 [+0.40, +1.05] |
| Concatenation augmentation, p=0.15, 2 to 4 pairs | None | Helps only inputs shaped like the augmentation | H2 not supported: E2 +0.06 [-0.19, +0.29], p=0.313; E2-synth +1.09 [+0.59, +1.68]; E1 non-inferiority met (lower bound -0.22) (`H2_*_seg_off.json`). `main` was trained with it before this result |
| Joint SentencePiece BPE, 16k, byte fallback | Unigram, BPE-dropout | Small vocabulary, longer sequences; no vocabulary or model-type sweep | sha256 `1fc208b5...`; no UNK or byte fallback on dev, test, E1, E2, E3 (`tokenizer_stats.json`) |
| WSD schedule: warmup 4,000, peak 7e-4, linear decay over the last 20% | Cosine, inverse-sqrt, not tried | Any stable checkpoint can start a cooldown (extension run) | `wsd_lr_scale`, `nmt/train.py:564-590`. Averaging 5 or 4 checkpoints did not beat the final one (objective 48.66, 48.69 against 48.75; `reports/final/main/selection.json`, no CI) |
| Beam 5, GNMT length penalty alpha 1.2, 3-gram block, segmentation above 192 source tokens | Greedy; T=64 | Alpha 1.2 was the top of the grid in all 6 tuning runs, so the optimum may be higher; segmentation moved `main` by at most +0.07 chrF | `selection.json`; segmentation tables in `reports/final/SUMMARY.md` |
| Selection on 0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1) | Selecting on dev or E3 | E2 is held out of the training pool but in-domain, so it flatters the winner | `nmt/selection.py` can load only E1 and E2; objective 48.7455 (`selection.json`) |
| MBR with chrF utility; log-probability ensembles | Single model, beam only | 8 or 16 candidates plus a pairwise chrF pass, slower than beam 5 | {{FINAL_SELECTED_CONFIG}}; objective {{FINAL_OBJECTIVE}} against v1 48.7455 |
| bf16 autocast, no loss scaling; label smoothing 0.1, dropout 0.1 | fp16 with GradScaler (the spec's T4 plan); stronger regularisation | Compute moved to an L4, which supports bf16; regularisation not tuned, `main` ran 8.86 epochs, below the 10-epoch trigger (`reports/epoch_accounting.json`) | `run_meta.json` `config.precision`; no skipped optimizer step in 24,645 (`run_audit.json`, `grad_skip_count` 0) |

## 2. References

1. Vaswani et al. 2017. Attention Is All You Need. NeurIPS 30. arXiv:1706.03762.
2. Su, Lu, Pan, Murtadha, Wen, Liu. 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
3. Xiong et al. 2020. On Layer Normalization in the Transformer Architecture. ICML 2020. arXiv:2002.04745.
4. Kasai, Pappas, Peng, Cross, Smith. 2021. Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation. ICLR 2021. arXiv:2006.10369.
5. Hägele et al. 2024. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024. arXiv:2405.18392.
6. Nguyen, Murray, Chiang. 2021. Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution. IWSLT 2021.
7. Sennrich, Zhang. 2019. Revisiting Low-Resource Neural Machine Translation: A Case Study. ACL 2019.
8. Koehn. 2004. Statistical Significance Tests for Machine Translation Evaluation. EMNLP 2004.
9. Press, Smith, Lewis. 2022. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR 2022. arXiv:2108.12409.

## 3. Challenges

**Concatenation padding.** Concatenation after bucketing padded each batch to its longest joined row: 80.6% of padded tokens were padding and micro-batches reached 131,584 tokens against a budget of 8,192 (PREREG 2026-10-02). I now draw the concat plan before bucketing (commit `ecb1a85`): 29.6% padding, largest micro-batch 8,192 (`reports/epoch_accounting.json`).

**Resume out-of-memory.** `torch.load` put the checkpoint back on the GPU and `train()` kept it, a second copy of weights and optimizer state (commit `dc6e7e8`). On the 8 GB RTX 3070 the S1 attempt ran out of memory at step 6 on all 8 resumes; the smoke model could not show it. Checkpoints now load to CPU, with a regression test.

**Environment.** `official/score.py` reads files in the platform encoding, cp1252 on Windows, so a file identical to the references scored BLEU 97.5 (README); the scorer is byte-pinned, so `nmt.evaluate.run_official_scorer` wraps it and sets `PYTHONUTF8=1`. The first Colab tag failed on Python 3.13; CI now runs 3.12 and 3.13.

**Process.** The 3070 was shared with another of my jobs: S1 waited 5.7 h and was stopped at step 3, so S1 to S3 moved to one Colab L4 session (PLAN.md, 2026-10-02). My commit `9e36831` blamed the wrong cause; I corrected the record in PLAN.md, not the history. PRs #25 and #26 merged into their stack bases, not main, and PR #27 carried the branch onto main; I now retarget every stacked PR to main first. COMET-22 was projected at 2.2 h on CPU (estimate, `HANDOFF.md`); I stopped after 5 of 7 chunks and kept the output marked incomplete (`reports/final/comet_partial_cpu_INCOMPLETE/`). COMET: {{FINAL_COMET_SUMMARY}}.

**Compute and effort.** About 8-9 hours of my hands-on time. Elapsed: 67.0 h from the first commit to the last on `main` (`0d6cce3`), 43.3 h from the first W&B run to the last run's final heartbeat (`reports/final/effort_compute.json`, `git_span`, `wandb_span`). The eight Colab L4 runs (pilot, `main`, S1 to S3, three extension runs) trained for 36,440.4 s, 10.12 h, about 15.6 CU at the 1.54 CU/h I reported (an ESTIMATE, not a Colab ledger figure): `main` 11,791.8 s, S1 to S3 1,785.1, 1,976.2 and 1,928.2 s (`reports/final/wandb_run_summaries.json`), extension 9,798.2 + 3,481.2 + 4,695.4 = 17,974.7 s (`reports/extension/ext_val_loss_summary.json`). The RTX 3070 added 48.1 s; its two pilots have no `train_wall_seconds` and are not added. Colab evaluation and final-selection hours: {{FINAL_COLAB_HOURS}} h.

## 4. Dev results by slice

Official scorer, 1,000-resample bootstrap 95% CIs. v1 is `main`/`final`, alpha 1.2, beam 5, T=192 (`reports/final/main/seg_tuned/eval.json`). Submitted model: {{FINAL_MODEL_NAME}}.

| Set | n | v1 BLEU | v1 chrF | Final BLEU | Final chrF |
|-------|----|----------|----------|--------|--------|
| Dev seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | {{FINAL_DEV_SEEN_BLEU}} | {{FINAL_DEV_SEEN_CHRF}} |
| Dev long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | {{FINAL_DEV_LONG_BLEU}} | {{FINAL_DEV_LONG_CHRF}} |
| Dev unseen | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | {{FINAL_DEV_UNSEEN_BLEU}} | {{FINAL_DEV_UNSEEN_CHRF}} |
| Dev OVERALL | 150 | 42.29 [38.72, 45.82] | | {{FINAL_DEV_OVERALL}} | |
| E1 | 1,940 | 35.98 [34.67, 37.30] | 54.98 [53.93, 56.11] | {{FINAL_E1_BLEU}} | {{FINAL_E1_CHRF}} |
| E2 | 1,000 | 38.02 [36.64, 39.39] | 61.78 [60.82, 62.71] | {{FINAL_E2_BLEU}} | {{FINAL_E2_CHRF}} |
| E2-synth | 300 | 37.73 [36.31, 39.07] | 63.31 [62.31, 64.29] | {{FINAL_E2SYNTH_BLEU}} | {{FINAL_E2SYNTH_CHRF}} |
| E3 | 1,000 | 19.45 [18.42, 20.52] | 42.31 [41.29, 43.27] | {{FINAL_E3_BLEU}} | {{FINAL_E3_CHRF}} |

Copy-the-source floor (output = the French source, same scorer and bootstrap; `reports/final/baseline_copy_source/eval.json`, `objective.json`):

| Set | n | Copy BLEU | Copy chrF |
|-------|----|----------|----------|
| Dev OVERALL | 150 | 16.10 [13.96, 18.74] | |
| E1 | 1,940 | 6.03 [5.10, 7.12] | 26.56 [25.83, 27.33] |
| E2 | 1,000 | 5.19 [4.34, 6.12] | 31.98 [31.52, 32.52] |
| E2-synth | 300 | 5.42 [4.46, 6.46] | 34.79 [34.22, 35.44] |
| E3 | 1,000 | 1.25 [0.95, 1.54] | 20.47 [20.00, 20.98] |

Selection objective: copy 18.8954, v1 48.7455. A word-level copy check flags 0.839 to 0.944 of words for the copy baseline, 0.024 to 0.124 for v1 and 0.058 to 0.176 for the references themselves (`calibration.json`); it cannot tell an untranslated word from a name or cognate.

The 60 and 30 sentence slices have wide intervals, so I draw conclusions from E1, E2 and E3. E2 comes from the training pool and E2-synth is built from E2, so neither tests unseen content. COMET-22: {{FINAL_COMET_SUMMARY}}.

## 5. Generalization gap

Pre-specified (spec.md section 10): OLS of sentence chrF on length, repetition, source rarity, dialogue punctuation and a domain flag over E1 and E3 (n=2,940), HC3 errors. The `main` E1-to-E3 chrF gap is 12.67; the domain flag carries +12.88 (101.6%) and the four covariates net -0.21; R-squared 0.178 (`reports/final/main/seg_tuned/analysis.json`, `gap_decomposition`). What I could measure explains none of the gap. Reference noise is about 0.22 chrF on E3 (`metric_artifact_share`). E3 chrF [41.29, 43.27] lies inside the dev unseen interval [40.74, 48.39], so E3 is a fair proxy.

Exploratory v2, post-hoc and not pre-registered (`reports/final/gap_v2/README.md`): Shapley shares of five heuristic feature groups. The explained share depends on the estimand, so I give a range: the five groups explain 8.7% in the primary model (OLS with a domain dummy, common slopes), 31.4% with pooled slopes and no dummy, 40.5% in an Oaxaca-Blinder split with E1 slopes and -7.5% with E3 slopes (`gap_shares.json`, `estimand_sensitivity_chrf_5groups`; point estimates, no CIs). In the primary model the residual (91.3% [82.7, 100.8]) includes the domain dummy and is not a measured domain effect; I do not interpret it. Target-side rarity contributes +2.34 chrF [+1.80, +2.99] and alignment -1.74 [-2.31, -1.23]; the other three groups are small. Caveats: heuristic features, linear common-slope form, one model, one reference per sentence; no causal claim.

## 6. What I would do next

Train a symmetric 6/6 baseline so the shallow-decoder claim has a number. Run ALiBi against RoPE myself. Sweep vocabulary size and model type. Add BPE-dropout. Tune alpha past 1.2 on a held-out set.

Extension (E1 validation loss, not a quality metric): the constant-LR run (to step 40,000) had its minimum, 2.9121, at step 38,000, ended at 2.9192 and was flagged by the overfit watch from step 31,500, which I read as a constant-LR plateau (an interpretation, untested). Branch A (decay to step 37,500) ended at its minimum, 2.8750; branch B (decay to 50,000) ended at its minimum, 2.8703, after flags at evaluations 45,000 to 46,000 and 48,000; `main` ended at 2.9041 (`reports/extension/ext_val_loss_summary.json`). Effect on E1 and E2: {{FINAL_SELECTED_CONFIG}}.

## 7. Links

Private repo: https://github.com/gaurav-gandhi-2411/fr-en-transformer. Public repo: {{LINK_REPO}}. Model: {{LINK_HF_MODEL}}. W&B: {{LINK_WANDB}}. Private HF repos (names only): `OWNER/fr-en-transformer-data`, `OWNER/fr-en-transformer-eval`.

Built with AI coding assistance; design, experiments and analysis are mine.
