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

## 6. Amendments (dated; each made before the results it governs; the last entry is made after the
ablation and main-run selection results exist and states what was already known)

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
- **2026-10-02 — ablation hardware, main-run selection candidates, dev disclosure.**
  Committed after the main run finished and before any selection or evaluation decode of it, and
  before any ablation training.
  - **(a) §3 hardware: ablations run on Colab L4. The RTX 3070 is retired.**
    - **Runs:** `configs/s1_sin_l4.yaml`, `s2_rope_l4.yaml`, `s3_rope_concat_l4.yaml`, W&B group
      `ablation_l4`, sequentially in one Colab session (notebook `CONFIG = "ablations_l4"`, tag
      v0.2.3-colab). All three share seed 1234, `planned_steps` 4107, micro-batch 4096,
      tokens/step 25,000, warmup 500 and bf16.
    - **Why retired:** the 3070 is shared with another of GG's workloads (intent-router). The first
      S1 attempt waited 5.7 h for the GPU, was stopped by contention at step 3, then hit a
      resume-memory bug (fixed in v0.2.2-colab). §3 forbids splitting S1/S2/S3 across hardware,
      so all three move.
    - **Consequences for §4 and §5:**
      - §4 (`main_ext_3070`) lapses: there is no 3070 run, so the main-run candidate is `main`
        only.
      - §5's "known validity limit" no longer applies: main and the ablations both trained in
        bf16 on an NVIDIA L4.
    - Everything else in §3 is unchanged: hypotheses, decision rules, primary segmentation-off
      decoding, and final-checkpoint comparison.
  - **(b) Main-run selection candidates** (checkpoints saved on Drive:
    step_00019000, 00020757, 00022500, 00024269, 00024645):
    1. `final`: step 24,645 alone.
    2. `avg_last5`: the mean of all five. This includes step 19,000, which is before decay.
    3. `avg_decay`: the mean of the decay-phase checkpoints 20,757, 22,500, 24,269 and 24,645.

    - **Decay start:** step 19,716 per the WSD config. `nmt/train.py:532-534` (`wsd_lr_scale`) sets
      `decay_len = round(planned_steps × cooldown_frac) = round(24,645 × 0.2) = 4,929` and
      `decay_start_step = planned_steps − decay_len = 19,716`; `cooldown_frac: 0.2` is in
      `configs/main.yaml`.
    - **Where 24,645 comes from:** it is not in `configs/main.yaml`, which keeps a 50,000
      placeholder. It reached the trainer through `--planned-steps 24645`: the notebook's
      `PLANNED_STEPS`, overriding the config at `nmt/train.py:1456-1457`. That value is the L4
      pilot's `plan.json`.
    - **Retention disclosure:** only five checkpoints exist because `prune_checkpoints`
      (`nmt/train.py:676-697`) protects decay-phase files only when `decay_start_step` is set,
      which happens only under `--cooldown-now`. The main run therefore kept its last
      `keep_last: 5` checkpoints, and `keep_decay_phase: true` had no effect. The decay-phase set
      above is the four decay-phase checkpoints that survived, not every one written.
    - **Selection:** the selection objective (§2) is unchanged and uses E1 + E2 only. Each
      candidate gets its own §1 decoding tuning. The candidate and decoding config with the
      highest objective are selected.
  - **(c) Disclosure: dev metrics logged during training.**
    - The main run's periodic in-training eval hook (every 500 steps) decoded the official dev set
      (all 150, per slice) together with fixed seeded subsets of E1 (500), E2 (300) and E3 (300).
      It decoded greedily (max length 1.5 × source + 10, 3-gram repeat block) and logged BLEU/chrF
      to W&B as training curves (`nmt/evaluate.py` `build_train_eval_fn` / `TrainEvalConfig`).
    - These curves were not used for any decision: no checkpoint, hyperparameter, decoding or
      stopping choice. The same holds for E3.
    - Official dev and E3 enter only the single final report of the selected model (§5). All
      selection uses E1 + E2.
- **2026-10-02 — post-selection amendment: α-grid extension, MBR, ensembling and an extension
  run (made AFTER the ablation and main-run selection results exist; disclosures first).**
  - **What was known when this was written:**
    1. The Colab selection summaries (from GG): `main` winner = `final`, objective 48.7455
       (BLEU 37.08, chrF 57.29, chrF-E1 54.98) at α = 1.2, beam 5, T = 192; ablation objectives
       S1 43.7781, S2 44.8382, S3 44.9363. **α = 1.2, the upper edge of the §1 grid, was the
       tuning winner in all 6 tuning runs** (3 `main` candidates, S1, S2, S3), on E1 + E2.
    2. The main run's training curves, including the E1 validation loss, whose minimum (2.9041)
       was at the last evaluation, so the run was not overfitting at its planned end.
    3. NOT yet looked at by the authors of this amendment: any official dev, E3, E2-synth,
       sacreBLEU, COMET or per-slice score of this eval, and the H1 / H2 tests (§3).
    4. The choices below were motivated by (1) and (2), which are selection-set results and
       training curves. No dev or E3 score informed them. The motivation is post-hoc and is not
       presented as pre-registered hypothesis testing; only the rules below are pre-registered.
  - **Fallback and safety net:** the `main` / `final` model at its tuned config (α 1.2, beam 5,
    T 192), uploaded at HF eval revision `c3d85982…`, is committed as the v1 submission and stays
    the submission unless a rule below replaces it.
  - **1. α grid.** For every candidate decoded under the rules below, α ∈ {1.2, 1.4, 1.6, 1.8,
    2.0} replaces the §1 grid (beam ∈ {1, 4, 5} and the T step unchanged), chosen by the §2
    objective on E1 + E2. If the winner is again at α = 2.0 it is reported as a grid-edge result,
    not extended further. §3's H1 / H2 are unchanged: their primary decoding is each ablation
    model's already-tuned config from the original grid. A re-tune of S1–S3 on the extended
    grid, if run, is a secondary sensitivity report only and never changes the H1 / H2 verdicts.
  - **2. MBR decoding.** Utility = sentence-level chrF exactly as computed by the vendored
    `official/score.py` (same n-gram order, β and tokenization). Pseudo-references = the
    candidate pool; the output is the pool member with the highest mean utility against the other
    members. Pools tried (tuned on E1 + E2 with the §2 objective, one per model/ensemble):
    beam n-best with N ∈ {8, 16} and/or epsilon sampling with ε = 0.02, N ∈ {8, 16}
    (sampling seed 1234). Beam settings inside a pool use the winner of rule 1 for that model.
  - **3. Ensembling.** Average per-step log-probabilities across models sharing the
    SentencePiece tokenizer (sha256 `1fc208b5…`). The ensembles considered are exactly: {main
    `final`, branch A}, {main `final`, branch B}, {branch A, branch B}, {main `final`, branch A,
    branch B}; single-member ensembles equal the single model. No other combinations.
  - **4. Extension training (WSD branching).** From `runs/main/ckpt/step_00019000.pt`, a
    stable-phase checkpoint (the WSD decay of the main run starts at step 19,716):
    - continue the stable phase to 40,000 steps at the stable learning rate, saving stable
      checkpoints at 30,000 and 40,000;
    - **Branch A:** linear decay from the step-30,000 stable checkpoint to 37,500;
    - **Branch B:** linear decay from the step-40,000 stable checkpoint to 50,000;
    - same data order continued, same hyperparameters (seed 1234, dropout 0.1, bf16, tokens per
      step, warmup unchanged), new run directories, W&B group `extend_l4`;
    - **Overfitting watch:** E1 validation loss is read at every evaluation; two consecutive rises
      above the running minimum flag the run. A flag is reported to GG, who decides whether to cut
      it. Absent a cut, branches train to their planned end.
  - **5. Final selection.** Candidates = {main `final`, branch A, branch B, the four ensembles of
    rule 3} × {beam with the rule-1 grid, MBR with the rule-2 pools}, on the §2 objective over E1 +
    E2. The highest objective is the submission, at its own tuned config; ties go to the
    earlier-listed candidate. Dev, E2-synth and E3 are report-only and decoded once for the winner
    (§5). For information only, not a gate, the paired-bootstrap Δ of the objective against the v1
    model (`nmt/compare.py --objective`, 1,000 resamples, seed 1234) is reported with its 95% CI.
    Note the winner is chosen on the same sets it is tuned on, so its E1 + E2 objective is
    optimistic; dev and E3 are the unbiased view.
  - **6. Production config.** Also reported: the best single model with beam search (no
    MBR / ensemble), with measured latency, on the same CPU/precision settings used for the
    benchmark.
  - **Unchanged:** §1 metrics and bootstrap, §2 objective, §3 hypotheses and decision rules,
    selection-set separation (dev / E2-synth / E3 never enter selection).

- **2026-10-03 — staged final selection (supersedes rule 5 and the beam grid of rule 1 of the
  2026-10-02 post-selection amendment; made before any extension, MBR or ensemble result
  exists).**
  - **Why:** rule 5 as written is an exhaustive search of 35 candidates (7 model sets × {1 beam
    + 4 MBR pools}). Its cost is about 20.6 h of Colab time (ESTIMATE from `python -m
    nmt.final_all estimate` with assumed decode rates, before the L4 rates were measured), and
    picking the maximum of 35 noisy E1 + E2 objectives inflates the winner's score (winner's
    curse). A staged search tests fewer candidates and reaches the same kind of answer.
  - **What was known when this was written:** the same as the 2026-10-02 amendment's list, plus
    the L4 benchmark of the `main` final model (greedy 41.097 sentences/s and 2,305.93 output
    tokens/s; beam 5 21.345 sentences/s and 1,198.86 output tokens/s). No extension, MBR or
    ensemble score exists. No dev, E3 or E2-synth score of any new candidate exists.
  - **Stage 1 (beam, 7 model sets).** Each of the 7 model sets (main `final`, branch A, branch
    B, and the four ensembles of rule 3) is tuned with beam search over α ∈ {1.2, 1.4, 1.6, 1.8,
    2.0} × beam ∈ {4, 5}, then the T step as in §1, all chosen by the §2 objective on E1 + E2.
    (Beam 1 is dropped from the extended grid.) The 2 model sets with the highest objective go
    to stage 2; ties go to the earlier-listed model set.
  - **Stage 2 (MBR, top 2 only).** The 4 MBR pools of rule 2 (beam n-best N ∈ {8, 16}; epsilon
    sampling ε = 0.02, N ∈ {8, 16}, seed 1234), on those 2 model sets only. Beam settings inside
    a pool use that model set's stage-1 winner α; T is re-tuned for MBR candidates as before.
  - **Final pick.** Among the 2 stage-1 beam winners and the 8 MBR configs (10 candidates), the
    highest §2 objective on E1 + E2 is the submission, at its own tuned config; ties go to the
    earlier-listed candidate (model-set order, beam before MBR).
  - **Reported alongside (not gates):**
    - paired bootstrap (`nmt/compare.py --objective`, 1,000 resamples, seed 1234) of the winner
      vs the runner-up, and of the winner vs the production config;
    - **production config** = the best single model (main `final`, A or B, by stage-1 objective)
      with beam search only, no MBR and no ensemble;
    - measured latency for the winner and for the production config, with the benchmark settings
      stated;
    - the winner's E1 + E2 objective remains optimistic, since it is chosen on those sets; dev and
      E3 are the unbiased view.
  - **Unchanged:** the §2 objective; dev, E2-synth and E3 stay report-only and are decoded once,
    for the winner; the v1 submission stays the fallback; rules 2, 3 and 4 of the 2026-10-02
    amendment (MBR definition, ensemble members, extension training) are unchanged.
- **2026-10-03 — clarification of the 2026-10-02 retention disclosure (c) above.** The
  retention bug (`keep_decay_phase` protecting nothing without `--cooldown-now`) was real, but
  the main run's decay-phase checkpoints did not suffer: with checkpoints roughly every 1,750
  steps (time-based, `ckpt_minutes` 15 in `reports/main_l4/run_meta.json`), four were written
  after the decay start at step 19,716 (20,757, 22,500, 24,269, 24,645) and all four survive.
  "Not every one written" in the disclosure should be read as "not every checkpoint of the run",
  not as a loss of decay-phase files. This is inferred from the file spacing; a Drive listing was
  not available to confirm it.

- **2026-10-03 — deadline fallback (made before any `final_all` result exists).**
  - Submission is due Sunday 2026-10-04, 17:00 IST. The `final_all` run (staged selection, §6
    amendments of 2026-10-02 and 2026-10-03) has not started when this is written; no
    extension-branch, MBR, ensemble or stage-1 score exists.
  - **Rule:** if `final_all` stage 2 has not completed by Sunday 2026-10-04 10:00 IST, the
    stage-1 winner (beam search, its stage-1 tuned config) is the submission. If `final_all`
    fails entirely, v1 (`main`, `final`, beam, the config recorded in the v1 submission) stands.
  - **Interim safety net:** `final_all` decodes, validates (330 ids, 0 empty) and uploads the
    stage-1 winner's test predictions to the private eval repo (`runs/final_all_stage1/`) BEFORE
    stage 2 starts, so the stage-1 fallback exists even if the session is interrupted.
  - Nothing else in the amendments changes: the §2 objective, selection on E1 + E2 only, and
    dev, E2-synth and E3 report-only.

- **2026-10-04 — closing note (made before any extension-branch, MBR, ensemble or stage-1 score
  exists).** `final_all` was not run within the deadline. Per §5 and the 2026-10-03 time-box, v1
  stands: `main`, `final`, alpha 1.2, beam 5, T 192 (`submission/test_predictions_v1.json`). The
  A6/A7 candidates (extension branches, ensembles, MBR, extended alpha grid) were trained or
  implemented but never scored, so no post-selection result exists and nothing was selected on one.
  They are reported as pre-registered, not evaluated.
- **2026-10-04 — precision and hardware correction.** The text above (preamble and §4 notes) plans
  fp16 on a Colab T4 for `main`. That plan was superseded before training: `main`, the pilot, the
  ablations S1 to S3 and the extension runs all trained in **bf16 on one Colab L4**
  (`reports/main_l4/run_meta.json` `config.precision`; no GradScaler, no skipped optimizer step in
  24,645). The original text is left as written; this note is the correction of record.
- **2026-10-04 — config correction.** The entry above stating that `configs/main.yaml` keeps a
  50,000 placeholder describes the config the main run used. The file now carries
  `planned_steps: 24645`, the value the run was given on the command line, so the one documented
  command resolves the same WSD schedule (decay start 19,716). No result changed.
