# Placeholders, references and open items for the three drafts

The drafts are `report/report.md`, `report/model_card.md` and `docs/WALKTHROUGH.md`. `python -m scripts.check_placeholders` lists every unresolved `{{NAME}}` and fails if one is missing from the table below. Add `--strict` once filling is done to fail on any that remain.

Resolved on 2026-10-03 and removed from the table: the gap v2 (`{{GAPV2_*}}`, from `reports/final/gap_v2/`), production benchmark (`{{PROD_*}}`, from `reports/final/production/`), extension (`{{EXT_*}}`, from `reports/extension/ext_val_loss_summary.json`, which is on PR #35 until it merges), the licence, the effort statement (`reports/final/effort_compute.json`, produced by `scripts/compute_effort.py`) and the three ablation wall times. The copy-source baseline rows come from `reports/final/baseline_copy_source/` (PR #37 until it merges). What remains depends on `final_all` (not yet run), COMET, the Colab hours and publication. The file and key names marked "confirm" must be checked against the real `final_all` output before filling. Every filled value must be copied from the named file, with its path and key kept in the commit message.

## 1. Placeholders

| Placeholder | Used in | What goes there | Intended source (path and key) |
|---|---|---|---|
| {{FINAL_MODEL_NAME}} | report, model card | Name of the submitted model, for example `ext_branch_b_l4/final` or `main+A ensemble` | `final_all` selection output, winner `members` (confirm path, expected `reports/final/final_all/`) |
| {{FINAL_MODEL_SHA256}} | model card | sha256 of the submitted weights; one per member for an ensemble | `final_all` manifest, `model/model.safetensors` sha256 (confirm) |
| {{FINAL_DECODE_CONFIG}} | model card | alpha, beam, segmentation threshold, and the MBR pool if any | `final_all` selection winner config (confirm key; v1 equivalent is `reports/final/main/selection.json` `winner`) |
| {{FINAL_SELECTED_CONFIG}} | report, walkthrough | One phrase: winning model set, beam or MBR pool, alpha, T | same as above |
| {{FINAL_OBJECTIVE}} | report | Selection objective of the winner on E1+E2 | same file, `winner.objective` |
| {{FINAL_TRAINING_DETAILS}} | model card | Steps, schedule and wall time of the winning member(s) if not `main` | `configs/ext_*_l4.yaml`; W&B group `extend_l4`; `train_wall_seconds_summary` as in `reports/main_l4/run_audit.json` |
| {{FINAL_DEV_SEEN_BLEU}} | report, model card | `value [lo, hi]` to 2 decimals | `reports/final/final_all/seg_tuned/eval.json` `sets.dev.official_ci_by_slice.bleu.seen` (`point`, `ci_low`, `ci_high`; path is confirm) |
| {{FINAL_DEV_SEEN_CHRF}} | report, model card | same | `...official_ci_by_slice.chrf.seen` |
| {{FINAL_DEV_LONG_BLEU}} | report, model card | same | `...official_ci_by_slice.bleu.long` |
| {{FINAL_DEV_LONG_CHRF}} | report, model card | same | `...official_ci_by_slice.chrf.long` |
| {{FINAL_DEV_UNSEEN_BLEU}} | report, model card | same | `...official_ci_by_slice.bleu.unseen_domain` |
| {{FINAL_DEV_UNSEEN_CHRF}} | report, model card | same | `...official_ci_by_slice.chrf.unseen_domain` |
| {{FINAL_DEV_ALL_BLEU}} | model card | same | `sets.dev.official_bleu_ci` |
| {{FINAL_DEV_ALL_CHRF}} | model card | same | `sets.dev.official_chrf_ci` |
| {{FINAL_DEV_OVERALL}} | report, model card | OVERALL `value [lo, hi]` | `sets.dev.overall_ci` |
| {{FINAL_E1_BLEU}} | report, model card | same | `sets.e1.official_bleu_ci` |
| {{FINAL_E1_CHRF}} | report, model card | same | `sets.e1.official_chrf_ci` |
| {{FINAL_E2_BLEU}} | report, model card | same | `sets.e2.official_bleu_ci` |
| {{FINAL_E2_CHRF}} | report, model card | same | `sets.e2.official_chrf_ci` |
| {{FINAL_E2SYNTH_BLEU}} | report, model card | same | `sets.e2synth.official_bleu_ci` |
| {{FINAL_E2SYNTH_CHRF}} | report, model card | same | `sets.e2synth.official_chrf_ci` |
| {{FINAL_E3_BLEU}} | report, model card | same | `sets.e3.official_bleu_ci` |
| {{FINAL_E3_CHRF}} | report, model card | same | `sets.e3.official_chrf_ci` |
| {{FINAL_COMET_SUMMARY}} | report, model card | COMET-22 system score and CI per set, or "not measured" | COMET column of the final report; `eval.json` `comet` once scored on GPU (confirm key). v1 has none: `reports/final/SUMMARY.md` says NOT MEASURED |
| {{FINAL_COLAB_HOURS}} | report | Colab hours of the `final_all` session | `final_all` run summary or the Colab Summary cell (confirm); CU only if Colab's panel shows it |
| {{LINK_REPO}} | report | Public repository URL | After the owner approves publication |
| {{LINK_HF_MODEL}} | report, model card | Hub model URL | After publication |
| {{LINK_WANDB}} | report | Public W&B run URL | After publication; the project is private today |

## 2. References, and how each was checked

Only papers already cited in `spec.md` or `PREREG.md` are listed. I checked bibliographic metadata (title, authors, year, venue) only; I did not re-read the papers, so every statement about what a paper concludes comes from `spec.md` and is UNVERIFIED (item U1 below).

| # | Reference as cited in the report | Cited in | Verified by |
|---|---|---|---|
| 1 | Vaswani et al. 2017, Attention Is All You Need, NeurIPS 30 | spec sections 5, 6 | arXiv API id 1706.03762 (8 authors, 2017-06-12); venue from the NeurIPS proceedings page `papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html` (title and authors match) |
| 2 | Su, Lu, Pan, Murtadha, Wen, Liu 2021, RoFormer, arXiv:2104.09864 | spec section 5 | arXiv API id 2104.09864 (6 authors, posted 2021-04-20). Crossref lists a journal version in Neurocomputing, DOI 10.1016/j.neucom.2023.127063, issued 2024-02; I kept the 2021 year because the spec says 2021 and the report cites the arXiv id |
| 3 | Xiong et al. 2020, On Layer Normalization in the Transformer Architecture, ICML 2020 | spec section 5 | arXiv API id 2002.04745 (10 authors, 2020-02-12), arXiv comment "Published on ICML 2020". Venue not checked against PMLR |
| 4 | Kasai, Pappas, Peng, Cross, Smith 2021, Deep Encoder, Shallow Decoder, ICLR 2021 | spec sections 5, 7 | arXiv API id 2006.10369 (5 authors, first posted 2020-06-18), arXiv comment "ICLR 2021 Final Version". The spec's 2021 is the venue year, not the posting year |
| 5 | Hägele et al. 2024, Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations, NeurIPS 2024 | spec section 6, `nmt/train.py` | arXiv API title search, id 2405.18392 (6 authors, 2024-05-28), arXiv comment "Spotlight at NeurIPS 2024". Proceedings page not checked |
| 6 | Nguyen, Murray, Chiang 2021, Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution, IWSLT 2021 | spec section 6 | ACL Anthology bib `2021.iwslt-1.33` (authors, IWSLT 2021 proceedings); Crossref DOI 10.18653/v1/2021.iwslt-1.33; arXiv API 2105.01691 |
| 7 | Sennrich, Zhang 2019, Revisiting Low-Resource Neural Machine Translation: A Case Study, ACL 2019 | spec section 4 | ACL Anthology bib `P19-1021` (ACL 2019, Florence, pages 211-221, DOI 10.18653/v1/P19-1021); arXiv API 1905.11901 |
| 8 | Koehn 2004, Statistical Significance Tests for Machine Translation Evaluation, EMNLP 2004 | spec section 8, PREREG section 1 | ACL Anthology bib `W04-3250` (EMNLP 2004, Barcelona, pages 388-395) |
| 9 | Press, Smith, Lewis 2022, Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, ICLR 2022, arXiv:2108.12409 | spec section 5 (named, no citation); report positional-encoding row | arXiv abs page `arxiv.org/abs/2108.12409` (citation meta: title, authors Press, Smith, Lewis; "Submitted on 27 Aug 2021, last revised 22 Apr 2022 (v2)"); OpenAlex API agrees on title and year 2021 (arXiv-only record); venue from the first-page header of the arXiv v2 PDF, "Published as a conference paper at ICLR 2022", read with pypdf. OpenReview and DBLP returned bot-check pages, and arXiv's API returned 503, so the ICLR proceedings listing itself was NOT checked |

Not listed on purpose:

- Szegedy et al. (label smoothing, cited in `nmt/train.py` only) and Hewitt et al. 2022 (epsilon sampling, cited in `nmt/mbr.py` only) are not in the spec or PREREG, so I left them out.
- Two arXiv ids I first guessed for Hägele and Nguyen were wrong and returned unrelated papers; the ids above come from title searches and the ACL Anthology, not from memory.

## 3. Open questions for the owner

All six earlier questions are RESOLVED on 2026-10-03 by the owner's decisions:

1. **Licence.** Code Apache-2.0; weights licence "other", research and evaluation use only; trained on OPUS-100 (licence not specified on its card); upstream corpus terms apply (model card, Licence section and YAML). Not done: there is no `LICENSE` file in the repository yet.
2. **Effort hours.** "About 8-9 hours of my hands-on time" (the owner's figure), plus the elapsed span and GPU hours computed in `reports/final/effort_compute.json`.
3. **Ablation wall times.** Filled from `reports/final/wandb_run_summaries.json`.
4. **Floor number.** Added: the copy-the-source baseline (`reports/final/baseline_copy_source/`, PR #37) is in the report and the model card.
5. **Stacked PR wording.** The report states only the facts the PR records show.
6. **Sponsor name and ALiBi.** The sponsor is not named in the report, model card or README (the README now says "a take-home challenge"). ALiBi is the rejected alternative in the positional-encoding row and reference 9, stated as not run by me.

Still open: `pyproject.toml` (`description`) and other tracked files outside this PR's scope may name the sponsor; `docs/internal/PUBLICATION_PLAN.md` (PR #36) covers the history rewrite.

## 4. UNVERIFIED statements

- U1. What each cited paper concludes (about 2x faster decoding from a deep encoder and shallow decoder; pre-LN stable with short warmup; RoPE extrapolation; WSD robustness to cutoffs; BPE size for low-resource data; concatenation helping low-resource translation) comes from `spec.md`. I verified metadata only.
- U2. "OPUS corpora carry mixed per-source licences" is from the brief. The dataset card does not say it.
- U3. RESOLVED (the three ablation wall times now come from `reports/final/wandb_run_summaries.json`).
- U4. The CU figures: 1.54 CU/h is the rate you reported (`reports/pilot_l4/pilot_summary.json`, `colab_pro_compute_units_per_hour`); 5.5 CU for `main` is `reports/main_l4/run_audit.json` `wall.cu_used_ESTIMATE`, an estimate; the new totals in `reports/final/effort_compute.json` are hours times 1.54 and are estimates, not a Colab ledger; they exclude Colab evaluation and `final_all` sessions, the two RTX 3070 pilots (no `train_wall_seconds`) and any time not logged to W&B. "First W&B run to last run" uses `heartbeat_at` of the latest run as the finish time (W&B has no `finished_at`). Elapsed git time is first to last author date reachable from `origin/main` at `0d6cce3` and will grow with later merges.
- U5. COMET CPU projections of 2.2 h and 4.5 h are quoted from `HANDOFF.md`; the projecting run's log is not in the repo.
- U6. That `tests/test_train_resume_memory.py` fails on the old code is taken from commit `dc6e7e8`'s message; I did not check out the old code to confirm.
- U7. How the concat-padding bug was discovered. `PREREG.md` gives the evidence (the epoch-accounting replay before the fix); the report says only that.
- U8. The description of the OPUS-100 mixture (subtitle-style dialogue, legal and patent text, web boilerplate) is read off evaluation examples, not a documented composition.
- U9. Venues of Xiong (ICML 2020) and Hägele (NeurIPS 2024) are taken from arXiv comments, not from the proceedings.
- U10. Loading the model by Hub repo id has not been run; nothing is published. Loading a local directory was run (model card, "How to use").
- U11. The 80.6% padding and 131,584-token micro-batch figures are from `PREREG.md` (2026-10-02). The pre-fix replay file is no longer in the repo, so I could not recompute them. The post-fix figures (29.6%, 8,192) I recomputed from `reports/epoch_accounting.json`.
- U12. Walkthrough claims about code behaviour were read from the code and the line ranges were checked by script; the behavioural claims (for example "a test scans the source") were not re-run as tests beyond the full suite passing.
- U13. Gap v2 is exploratory and post-hoc; its figures are copied from `reports/final/gap_v2/` (README, `gap_shares.json`), with the "about 21%" NLL share taken from its README, not recomputed here. The production benchmark's p50 and throughput figures are copied from its README tables, which its README says are copied from `results.json`; I re-read the two p50 values (287.8 and 481.4 ms) in `results.json` only.
- U14. The model card usage snippet was run on CPU on 2026-10-03 against a manifest-verified pull of `runs/main/model` (`verified: True`, `model_files_checked: True`) with two sentences of my own; loading by Hub repo id has not been run.
