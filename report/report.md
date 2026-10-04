# French-to-English Transformer from scratch: decisions, results, generalization

Gaurav Gandhi · 4 October 2026

Dev OVERALL is 42.29 [38.72, 45.82] against 16.10 for copying the source; long inputs hold up (dev long chrF 63.16); literature (E3) is 12.67 chrF below the in-domain proxy.

I trained a 50.2M-parameter encoder-decoder from scratch on OPUS-100 en-fr, with no pretrained model, tokenizer or language-ID tool. The submitted model ("v1") is run `main`, checkpoint `final`, decoded with beam 5, GNMT length penalty alpha 1.2, 3-gram block and segmentation above 192 source tokens. Every number except my hands-on hours comes from a file in the repository. Result paths are relative to `reports/`; other paths are from the repository root.

## 1. Architecture decisions and why

| Choice | Rejected alternative | Trade-off and evidence |
|---|---|---|
| Encoder-decoder [1] | Decoder-only (prefix LM) | The encoder reads the whole source bidirectionally and cross-attention gives an explicit alignment, the standard sample-efficient shape for 0.9M pairs. I did not train a decoder-only model, so this is a prior, not a measurement |
| 8 encoder / 4 decoder layers, d=512, 8 heads (head dim 64), FFN 2048, pre-LN [3], GELU, tied embeddings: 50,229,248 parameters (`main_l4/run_meta.json`) | Transformer-big; symmetric 6/6 | A shallow decoder makes autoregressive decoding cheaper [4]. I trained no symmetric baseline, so the speed gain is the paper's, not mine. Size set by the budget: 24,645 steps (8.86 epochs, `epoch_accounting.json`) took 3.3 h on one L4 |
| Attention written by hand on `F.scaled_dot_product_attention`, KV cache for decoding | `torch.nn.Transformer` | `nn.MultiheadAttention` has no hook to rotate queries and keys (RoPE) and `nn.Transformer` has no incremental KV cache, so every beam step would recompute the prefix |
| RoPE [2], 4,107-step ablation against sinusoidal | Sinusoidal; ALiBi [9], cited and not run | H1 supported: E2 chrF +0.89 [+0.66, +1.14], E2-synth +3.11 [+2.40, +3.85] (`final/compare/H1_*_seg_off.json`). The ablation is 1.8 epochs, so its transfer to the 24k-step run is assumed, not shown |
| Concatenation augmentation [6], p=0.15 (2 to 4 pairs) | None | H2 not supported (Section 5). `main` was trained with it before that result |
| Joint SentencePiece BPE, 16k, byte fallback | Separate vocabularies, Unigram, BPE-dropout | A small joint vocabulary suits 0.9M pairs [7], copies names and allows three-way tying. UNK and byte-fallback rate 0 on dev, test, E1, E2, E3 (`tokenizer_stats.json`). No vocabulary sweep |
| WSD schedule: warmup 4,000, peak 7e-4, linear decay over the last 20% [5] | Cosine, inverse-sqrt | Any stable checkpoint can start a cooldown. Averaging the last 5 checkpoints (includes step 19,000, before decay) or the 4 decay-phase ones did not beat `final` (objective 48.66, 48.69 against 48.75; `final/main/selection.json`) |
| bf16 autocast, label smoothing 0.1, dropout 0.1 | fp16 with loss scaling | The L4 supports bf16; no skipped optimizer step in the 492 logged rows to step 24,600 (`main_l4/run_audit.json`). Regularisation not tuned |
| Beam 5, GNMT alpha 1.2, 3-gram block; sources above 192 tokens split at sentence ends; output never empty (beam, then greedy, then source copy) | Greedy; T=64 | Alpha 1.2 was the top of the grid in all 6 tuning runs, so the optimum may be higher. Segmentation moved `main` by at most +0.07 chrF (`final/main/seg_*/eval.json`) |
| Selection on 0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1) | Selecting on dev or E3 | `nmt/selection.py` loads only E1 and E2; v1 objective 48.7455 |

## 2. Paper references

[1] Vaswani et al. 2017. Attention Is All You Need. NeurIPS. arXiv:1706.03762. [2] Su et al. 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864. [3] Xiong et al. 2020. On Layer Normalization in the Transformer Architecture. ICML. arXiv:2002.04745. [4] Kasai et al. 2021. Deep Encoder, Shallow Decoder. ICLR. arXiv:2006.10369. [5] Hägele et al. 2024. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS. arXiv:2405.18392. [6] Nguyen, Murray, Chiang. 2021. Data Augmentation by Concatenation for Low-Resource Translation. IWSLT. [7] Sennrich, Zhang. 2019. Revisiting Low-Resource Neural Machine Translation. ACL. [8] Koehn. 2004. Statistical Significance Tests for Machine Translation Evaluation. EMNLP. [9] Press, Smith, Lewis. 2022. Train Short, Test Long (ALiBi). ICLR. arXiv:2108.12409.

## 3. Challenges and resolutions

- **Concatenation padding:** padding was 80.6% of padded tokens with micro-batches up to 131,584 tokens against a budget of 8,192; drawing the concat plan before bucketing gave 29.6% and a maximum of 8,192 (before: `PREREG.md` 2026-10-02 entry; after: `epoch_accounting.json`; commit `ecb1a85`).
- **Scorer encoding:** `official/score.py` reads files in the platform encoding (cp1252 on Windows), so the dev references used as predictions scored BLEU 97.50 (reproduced); the scorer is byte-pinned, so my wrapper sets `PYTHONUTF8=1` and the same file scores 100.
- **Checkpoint-load OOM:** `torch.load` put a second copy of weights and optimizer state on the GPU; on the 8 GB RTX 3070 all 8 resumes of one run died at step 6. Checkpoints now load to CPU, with a regression test (commit `dc6e7e8`).
- **Python 3.13 on Colab:** the first Colab tag failed on Colab's 3.13; the notebook now keeps Colab's own CUDA torch via a constraints file and CI runs 3.12 and 3.13, executing the notebook in smoke mode.
- **GPU contention:** an ablation waited 5.7 h for a shared GPU and was preempted at step 3; I moved all ablations to one Colab L4 session.
- **Retention bug:** `keep_decay_phase` protected nothing without `--cooldown-now`, so `main` kept only its last 5 checkpoints; all 4 decay-phase files written survive (inferred from file spacing; disclosed in `PREREG.md`).
- **Reproducibility:** pinned `uv.lock` and Colab requirements, torch pinned by constraints, seed 1234 everywhere, Colab runs from git tags, and a resume test that matches the saved loss trajectory.

## 4. Dev results by slice

Official scorer, 1,000-resample bootstrap 95% CIs [8] (`final/main/seg_tuned/eval.json`). The copy-the-source floor outputs the French source (`final/baseline_copy_source/eval.json`). E1, E2 and E3 are held-out proxies built from public data (Section 7); E2-synth is 300 synthetic inputs of 400 to 900 French characters made by joining 2 to 4 E2 pairs.

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

The 30 and 60 sentence slices have wide intervals, so I draw conclusions from E1, E2 and E3. E2 comes from the training pool and E2-synth is built from E2, so neither tests unseen content. E3 chrF (42.31) lies inside the dev-unseen interval [40.74, 48.39], consistent with it being a reasonable proxy. COMET was not measured.

**Where it fails** (`final/main/seg_tuned/diagnostics.json`, `eval.json`). By source length, chrF is lowest for short inputs (48.16 at 10 words or fewer, 60.89 at 41 to 80, 60.32 above 80; E1+E2+E3 pooled). By source rarity, chrF is 55.19 and 56.05 in the two most common quintiles and 50.45 in the rarest. Failure rates on E1: truncation (hypothesis under half the reference length) 2.2%, overlong (over 1.5x) 4.0%, repeated 3-gram 0.5%; on E2 the repeated-3-gram rate is 3.0% but the references have 14.7%, so repetition is mostly legitimate. A strict untranslated-copy rate is 0.0% in every set.

## 5. Generalization evidence

Seen to unseen (dev): chrF 50.08 to 44.76 (-5.32) and BLEU 32.43 to 21.99 (-10.43), intervals overlapping. On the large proxies, E1 to E3 is -12.67 chrF and -16.52 BLEU, outside the intervals. Long inputs do not degrade: dev long 63.16 and E2 61.78 chrF, above their seen counterparts, though E2 is in-domain.

- **H1 (RoPE over sinusoidal): supported**, Section 1.
- **H2 (concatenation, S3 over S2): not supported.** E2 chrF +0.06 [-0.19, +0.29], p=0.313, so the pre-registered E2 criterion fails; E2-synth +1.09 [+0.59, +1.68]; E1 non-inferiority met (lower bound -0.22) (`final/compare/H2_*_seg_off.json`).

**Gap decomposition.** The pre-registered OLS of sentence chrF on length, repetition, source rarity, dialogue punctuation and a domain flag (E1 and E3, n=2,940, HC3) leaves the E1-to-E3 chrF gap of 12.67 almost entirely with the domain flag (+12.88); the four covariates net -0.21 (`final/main/seg_tuned/analysis.json`). An exploratory, post-hoc v2 with five heuristic feature groups gives a share explained that depends on the estimand: 8.7% primary (common slopes with a domain dummy), 31.4% pooled slopes without a dummy, 40.5% Oaxaca-Blinder with E1 slopes and -7.5% with E3 slopes; point estimates without CIs (`final/gap_v2/gap_shares.json`). In the primary model the residual (91.3%) includes the dummy and is not a measured domain effect. **Untested hypotheses:** the gap reflects literary vocabulary and style absent from OPUS-100, and rare target-side words drive it (target-side rarity +2.34 chrF [+1.80, +2.99]); neither was tested by an intervention. Reference noise is about 0.22 chrF on E3.

## 6. What I would do next

Run `final_all`: the extension branches (E1 validation loss 2.904 for `main` (`main_l4/run_audit.json`), 2.875 and 2.870 for branches A and B (`extension/ext_val_loss_summary.json`)), log-probability ensembles, MBR with a chrF utility and an extended alpha grid, since 1.2 was the grid edge. These are implemented and pre-registered in `PREREG.md` but not evaluated within the deadline, so no post-selection result exists and v1 stands. Then back-translation, BPE-dropout and R-Drop, a larger model, and several seeds.

## 7. Data and constraints

OPUS-100 en-fr (revision `805090dc`): 1,000,000 raw pairs; 37,122 duplicates, 13,061 with French equal to English, 13,116 outside the length ratio [1/3, 3], 10,412 with over 50% non-letters, 2,540 leakage-guard removals (exact and near-duplicate matches to dev and test sources, dev references, E1 and E3) and 1,001 for the E2 holdout leave 922,748; 1,078 pairs over 256 tokens leave **921,670 (92.2%)** (`data/data_manifest.json`, `tokenizer_stats.json`, `epoch_accounting.json`). The guard exists because OPUS-100 contains near-copies of the evaluation sentences; the post-check finds 0 hits. E1 is the OPUS-100 validation split (1,940 after leakage removal), E2 is 1,000 long pairs held out of training, E3 is 1,000 `opus_books` pairs used for reporting only: `opus_books` was never trained on, tuned on or selected on. Trained from scratch on one NVIDIA L4 (24 GB), bf16, seed 1234, no paid API. The 24,645 planned steps are in `configs/main.yaml`, from the L4 pilot (`pilot_l4/pilot_summary.json`).

## 8. Effort and compute

About 8 to 9 hours of my hands-on time over about three days; 10.1 GPU-hours of training on one L4 (main 3.3 h, ablations 1.6 h, extension 5.0 h, pilot 0.3 h, each rounded; `final/effort_compute.json`).

Built with AI coding assistance; design, experiments and analysis are mine.

Links: code https://github.com/gaurav-gandhi-2411/fr-en-transformer, model https://huggingface.co/gauravgandhi2411/fr-en-transformer, W&B https://wandb.ai/gauravgandhi429-gaurav-gandhi/fr-en-transformer-public. Reproduce: `python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234` (one L4, about 3.3 h of training).
