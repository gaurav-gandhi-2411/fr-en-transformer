# Overfit-watch lines of the extension run: what they meant

Scope: W&B group `extend_l4` (entity `gauravgandhi429-gaurav-gandhi`, project `fr-en-transformer`),
runs `ext_stable_l4` (86e3ddfa), `ext_branch_a_l4` (ba8c6bf4), `ext_branch_b_l4` (1f7d5761).
All numbers below come from `reports/extension/ext_val_loss_summary.json`, produced from a
read-only W&B API pull (retrieved 2026-10-03T02:31:56Z; raw dump sha256 in that file's
`provenance`).

## Mechanism (a reporting artefact, not a training-side bug)

Two places turn a per-evaluation flag into a "run-level" statement, and both were sticky:

1. `colab/train.ipynb`, `extend_watch_lines` (before this PR) printed
   `FLAGGED since step <first flagged step> (rises=<rises at the LAST eval>)` whenever ANY eval row
   in `metrics.jsonl` had `overfit_flag == 1`. "Since" was the first flagged step, but the flag
   was never required to still hold at the end, and `rises` was taken from the final row. A run that
   was flagged mid-way and then recovered therefore printed "FLAGGED since 45000 (rises=0)".
2. `nmt/train.py` set the W&B summary `overfit_flag = True` (and `overfit_flag_first_step`) the first
   time the flag fired and never cleared it, so the summary meant "ever flagged".

The training-time logic itself, `OverfitWatch.update` (`nmt/train.py`), is per evaluation and not
sticky: `flag = consecutive_rises >= 2`, where a value `<=` the running minimum resets the streak.
The per-eval row fields `eval/overfit_flag`, `eval/val_loss_running_min`, `eval/val_loss_rises` are
the CURRENT state at that evaluation.

Evidence that the training-side flags are correct: for all three runs I replayed the W&B
`eval/val_loss` series through `OverfitWatch` (fresh minimum at the run's first evaluation) and
compared with the logged `eval/overfit_flag` and `eval/val_loss_rises` per evaluation. Keys
`recomputed_flag_matches_logged` and `recomputed_rises_matches_logged` are `true` for all three runs
in `ext_val_loss_summary.json`. The watch never stops or alters training (it only prints, writes
the eval row and the W&B summary), so none of the training numerics (optimizer, data order, RNG, LR,
checkpoints) are affected by either the old reporting or this fix.

## What each run's watch really said

| run | evals (steps) | flagged evals (steps) | state at final eval | last / min val_loss |
|---|---|---|---|---|
| ext_stable_l4 | 42 (19,500 to 40,000) | 31,500; 34,500; 37,000; 37,500; 39,000; 39,500; 40,000 | FLAGGED, 4 consecutive rises, last > min | 2.9192 / 2.9121 (step 38,000) |
| ext_branch_a_l4 | 15 (30,500 to 37,500) | none | not flagged, 0 rises, last == min | 2.8750 / 2.8750 (step 37,500) |
| ext_branch_b_l4 | 20 (40,500 to 50,000) | 45,000; 45,500; 46,000; 48,000 | not flagged, 0 rises, last == min | 2.8703 / 2.8703 (step 50,000) |

W&B summary fields as they were stored (old semantics, sticky): stable `overfit_flag=True`,
`overfit_flag_first_step=31500`; branch A neither field set; branch B `overfit_flag=True`,
`overfit_flag_first_step=45000`. So branch B's "FLAGGED since 45000 (rises=0)" is explained: it
was flagged at 45,000 to 46,000 and again at 48,000, recovered to a new minimum at 48,500
(2.8755), and ended at its minimum with 0 rises. The stable run's line was accurate on both counts
(still flagged at the end, rises=4).

GG's numbers checked against the data: every figure given (stable min 2.9121 at 38,000, flagged
since 31,500, rises=4; A last = min = 2.8750; B last = min = 2.8703, flagged since 45,000) matches
the series. No correction needed.

## Statement for the report

The stable-phase flag reflects the high-LR plateau: `configs/ext_stable_l4.yaml` holds the LR
constant at 7.0e-4 from the init step to step 40,000, and E1 val loss there moves by a few
thousandths between evaluations (rises of up to about 0.010 around a slowly falling trend, from 2.9841
at step 19,500 to a minimum of 2.9121 at step 38,000). The "constant-LR noise" reading is an
interpretation consistent with this series, not something these runs test directly. Both decayed
branches end at their own minima: Branch A 2.8750 at step 37,500 (decay 30,000 to 37,500) and
Branch B 2.8703 at step 50,000 (decay 40,000 to 50,000), versus the stable run's own minimum of
2.9121 and its values of 2.9198 at step 37,500 and 2.9192 at step 40,000 at the matching steps.

## Run bookkeeping (from W&B summary / config)

| run | steps run (init -> final) | train_wall_seconds (summary) | init checkpoint (config `init_from`) |
|---|---|---|---|
| ext_stable_l4 | 19,000 -> 40,000 | 9798.18 | `/content/drive/MyDrive/fr-en-transformer/runs/main/ckpt/step_00019000.pt` |
| ext_branch_a_l4 | 30,000 -> 37,500 | 3481.18 | `/content/drive/MyDrive/fr-en-transformer/runs/ext_stable_l4/ckpt/step_00030000.pt` |
| ext_branch_b_l4 | 40,000 -> 50,000 | 4695.37 | `/content/drive/MyDrive/fr-en-transformer/runs/ext_stable_l4/ckpt/step_00040000.pt` |

Final checkpoint paths: UNVERIFIED. The task text said GG listed them, but no list reached this
session, and W&B does not record them. By the repo's naming they would be
`.../runs/<run>/ckpt/step_<final>.pt` (stable step_00040000, A step_00037500, B step_00050000);
that is an inference, to be confirmed against Drive.

## What changed in the reporting

- Notebook line now reads, e.g. for branch B:
  `flag history: first flagged at step 45000, last flagged at step 48000 (4 flagged evals); current state at the final eval: not flagged (consecutive rises 0, last == min)`.
- W&B summary now carries `overfit_flagged_ever` and `overfit_flag_current` (updated at every
  evaluation), plus `overfit_flag_last_step`; legacy `overfit_flag` and `overfit_flag_first_step`
  keep their old "ever" meaning. The three finished runs' stored W&B summaries are NOT rewritten
  (read-only access); they keep the old sticky fields.
