# PREREG.md — pre-registered analysis plan

Committed before any GPU training run (ablations on the local RTX 3070, pilot/main on Colab T4).
Anything decided after results are seen is logged in PLAN.md as **post-hoc** and never presented
as pre-registered.

## 1. Fixed conventions (apply to every test below)

- **Metric implementations:** official `score.py` BLEU and chrF (vendored, byte-identical).
  sacreBLEU and COMET-22 are reported alongside, never used for a decision.
- **Paired bootstrap:** Koehn (2004), `nmt/compare.py`, 1,000 resamples, seed 1234, the same
  resample indices for both systems. Δ = A − B, a 95% percentile CI, and a one-sided p equal to the
  fraction of resamples with Δ ≤ 0. "Significant" means p < 0.05.
- **Eval sets:**

  | Set | n | Role |
  |---|---|---|
  | E1 seen-proxy | 1,940 | Selection/tuning, and the seen non-inferiority check |
  | E2 long-proxy | 1,000 | Tuning, and long-input evidence |
  | E2-synth (synthetic) | 300 | Evaluation only; 100 each at 400–600 / 600–800 / 800–900 French characters, concatenations of 2–4 E2 pairs |
  | Official dev | 150 | 60 seen / 30 long / 60 unseen; reported with CIs, never used for decisions (too small) |
  | E3 books-proxy | 1,000 | Reported only |

  E2-synth reuses E2 sentences, so it tests length generalization, not held-out content.
- **Decoding tuning** (per checkpoint, `nmt/tune.py`, full E1 + E2, no prefixes):
  - α ∈ {0.6, 0.8, 1.0, 1.2} × beam ∈ {1, 4, 5}, chosen by the selection objective.
  - Then the segmentation threshold T ∈ {64, 128, 192, 256, off}, chosen on E2 with E1 held fixed.

## 2. Selection objective

`objective = 0.4 · BLEU(E1 ∪ E2) + 0.4 · chrF(E1 ∪ E2) + 0.2 · chrF(E1)`

This mirrors the official OVERALL, with E1 standing in for the unseen slice
(`nmt/selection.py:selection_objective`). It is computed only on E1 and E2. E3, the official dev
set and E2-synth can never enter selection: this is enforced structurally and by tests.

## 3. Ablation hypotheses (local RTX 3070, W&B group `ablation_3070`)

S1, S2 and S3 share data shards (HF revision `c40e3937`), seed 1234, tokens per step, warmup,
`planned_steps` (derived once from `pilot_3070` for a 40-minute budget) and precision (bf16).
Each run must complete its full WSD decay to the same `planned_steps`. A run cut short by the
wall-clock safety cap invalidates the comparison, and it is rerun rather than analysed. Each
comparison uses the final checkpoint (single, not averaged).

**Primary decoding for H1 and H2:** each model's own tuned α and beam with **segmentation off**,
because segmentation would hide the length effect under test. Results at each model's full tuned
config (including T) are reported as secondary.

### H1: RoPE vs sinusoidal (S2 vs S1)

- **Hypothesis:** RoPE improves chrF on long inputs.
- **Δ:** chrF(S2) − chrF(S1).
- **Decision:** H1 is supported iff Δ > 0 with p < 0.05 on **both** E2 and E2-synth (pooled
  over its three buckets). This is an intersection-union test, so no multiplicity correction is
  needed.
- **Also reported, not decisive:** dev `long` (n = 30, underpowered), each E2-synth bucket, the
  >80-word bucket over E1 + E2 + E3, and BLEU.

### H2: concat augmentation (S3 vs S2)

- **Hypothesis:** concat augmentation helps long inputs without hurting seen inputs by more than
  0.5 chrF.
- **Δ:** chrF(S3) − chrF(S2).
- **Decision:** H2 is supported iff **all** of the following hold:
  1. Δ > 0 with p < 0.05 on E2.
  2. Δ > 0 with p < 0.05 on E2-synth (pooled).
  3. Seen non-inferiority: the lower bound of the 95% CI of Δ chrF on E1 is > −0.5.
- **Also reported, not decisive:** dev `seen` (n = 60) and dev `long` (n = 30), each with a CI.

Each result is stated as: hypothesis → Δ [95% CI], p → supported or not.

## 4. Inclusion rule for the extended run (`main_ext_3070`)

`main_ext_3070` is `main` with about 2× the planned tokens, trained on the RTX 3070. It is **not
run without GG's approval**.

- If run, it becomes a submission candidate only if, after its own decoding tuning, a paired
  bootstrap of the **selection objective** over E1 + E2 shows it beats the Colab `main` model:
  - Δ = objective(ext) − objective(main);
  - E1 and E2 resampled stratified (`nmt/compare.py --objective`), 1,000 resamples, seed 1234;
  - one-sided p < 0.05.
- Otherwise `main` stands.

## 5. Submission rule

- Submit the eligible candidate with the highest selection objective, each at its own tuned
  decoding config. The candidates are Colab `main`, and `main_ext_3070` only if it passed §4.
- Official dev and E3 are evaluated **once**, for the chosen model and config, and never
  influence the choice.
- The model card and report state the training hardware (GPU name), precision (fp16 on T4, bf16
  on RTX 3070), torch/CUDA versions, git SHA (tag) and HF data revision of the submitted model.
- Known validity limit: the ablations ran in bf16 on an RTX 3070 while `main` runs in fp16 on a
  T4. Ablation conclusions are assumed, not shown, to transfer across precision and hardware.

## 6. Amendments (dated; each made before any ablation training run)

- **2026-10-01 — leakage near-duplicate rule kept (GG decision).**
  - **What stays:** the guard keeps removing train pairs whose normalized fr or en is a
    near-duplicate (lowercase, alphanumeric-only key) of any dev/test/E-set sentence. This
    matches the data the v0.2-colab shards were built from; no shard is rebuilt.
  - **Evidence** (`reports/audit/leakage_audit.json`):
    - 85.70% of near-dup removals are 1–3-word sentences: 2,139 of 2,496 pairs.
    - That is 0.231% of the 926,289 pairs entering the guard.
  - **Rationale:** eval contamination outweighs the negligible coverage loss.
- **2026-10-01 — ablation operations (not analysis).**
  - Ablation runs checkpoint every 5 minutes. A run interrupted by GPU contention or OOM resumes
    from its last checkpoint, keeping the identical step count (2,889), seed and data order.
  - Resumes are recorded (`resume_count`, wait time, redone steps). Pure training wall time is
    reported separately from wait time.
  - None of this changes §1–§5.
- **2026-10-02 — concatenation augmentation fixed before any S3 or main training.**
  - **Bug:** concat augmentation was applied after bucketing and padded whole batches to the
    concatenated length. At micro-batch 4096 / concat 0.15, 80.6% of padded tokens were padding
    and micro-batches reached 131,584 tokens, against a budget of 8,192 padded src+tgt tokens.
    Evidence: the `reports/epoch_accounting.json` replay before the fix.
  - **Fix:** the per-epoch concat plan is now drawn before bucketing, so every micro-batch
    respects its budget (29.6% padding, max 8,192). With concat off, batches are unchanged, so
    S1/S2 data order is identical.
  - **Effect on the hypotheses:** H2 (S3 vs S2) is evaluated on the fixed implementation. No S3
    or main step has run on the old one.
- **2026-10-02 — data-limited regularization amendment NOT triggered.**
  - The trigger was "main > 10 epochs". On the trainer's own token basis (padded src+tgt; see
    `nmt/data/loader.py` `batch_token_count`), main = 24,645 steps / 2,782 steps per epoch =
    8.86 epochs (micro-batch 4096, concat 0.15, seed 1234; `reports/epoch_accounting.json`).
  - The main config is therefore unchanged: dropout 0.1, label smoothing 0.1, existing
    retention and selection.
