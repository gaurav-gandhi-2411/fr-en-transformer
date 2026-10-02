# Final evaluation of the 4 runs

Every number below is printed from a JSON in this directory (named under each table). Scores are official `score.py` BLEU/chrF (decision metrics) with 95% bootstrap CIs; sacreBLEU and COMET-22 are reported alongside and never used for a decision (PREREG §1). Each model is shown at its own tuned decoding config (the config it would ship with); the segmentation-OFF numbers are in the seg-off section.

**COMET-22: NOT MEASURED in this report.** The local CPU run was stopped because its projected time exceeded the 2 h budget; COMET moves to a GPU session. A partial CPU run is kept, clearly marked, in `comet_partial_cpu_INCOMPLETE/` (not used in any table here).

## Selection check (HF `selection.json`)

| run | winner | objective | alpha | beam | T |
|---|---|---|---|---|---|
| main | final | 48.7455 | 1.2 | 5 | 192 |
| s1_sin_l4 | final | 43.7781 | 1.2 | 4 | 64 |
| s2_rope_l4 | final | 44.8382 | 1.2 | 5 | 64 |
| s3_rope_concat_l4 | final | 44.9363 | 1.2 | 5 | 64 |

Source: `<run>/selection.json` (verbatim copy of the HF `runs/<run>/selection.json`; `source/` is gitignored via the repo's `runs/` rule, re-pull with `scripts.eval_local`).

## Per-model results (tuned decoding config)

### main (deep-enc/shallow-dec, 24,645 steps)

alpha 1.2, beam 5, segmentation T=192; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 32.77 [26.98, 38.05] | 50.57 [46.86, 54.39] | - | 30.91 / 54.45 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 42.29 [38.72, 45.82] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | - | - |
| E1 seen-proxy | 1940 | 35.98 [34.67, 37.30] | 54.98 [53.93, 56.11] | - | 33.84 / 57.43 |
| E2 long-proxy | 1000 | 38.02 [36.64, 39.39] | 61.78 [60.82, 62.71] | - | 36.12 / 61.43 |
| E2-synth (synthetic) | 300 | 37.73 [36.31, 39.07] | 63.31 [62.31, 64.29] | - | 35.75 / 62.67 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 37.78 [35.01, 40.40] | 62.85 [60.93, 64.82] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 36.28 [33.90, 38.53] | 62.19 [60.44, 63.84] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 38.91 [36.49, 41.17] | 64.88 [63.40, 66.35] | - | - |
| E3 books-proxy | 1000 | 19.45 [18.42, 20.52] | 42.31 [41.29, 43.27] | - | 18.76 / 42.83 |

Source: `main/seg_tuned/eval.json`.


Length buckets (E1+E2+E3 pooled):

| source words | n | BLEU [95% CI] | chrF [95% CI] |
|---|---|---|---|
| <=10 | 1152 | 30.54 [28.52, 32.59] | 48.16 [46.53, 49.79] |
| 11-20 | 685 | 29.53 [27.37, 31.84] | 49.98 [48.46, 51.54] |
| 21-40 | 1230 | 31.41 [30.14, 32.67] | 55.23 [54.22, 56.17] |
| 41-80 | 795 | 35.98 [34.70, 37.32] | 60.89 [59.93, 61.92] |
| >80 | 78 | 37.45 [32.91, 40.94] | 60.32 [57.15, 63.59] |

Source: `main/seg_tuned/eval.json` (`length_buckets_e1_e2_e3`).


### S1 sinusoidal (4,107 steps)

alpha 1.2, beam 4, segmentation T=64; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 27.19 [21.92, 32.23] | 46.35 [42.87, 49.67] | - | 25.22 / 50.40 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 37.52 [34.00, 40.58] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 31.90 [22.37, 41.74] | 59.17 [53.32, 65.05] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 27.28 [20.10, 34.26] | 45.79 [39.47, 52.24] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 17.81 [14.00, 21.53] | 40.50 [36.75, 44.18] | - | - |
| E1 seen-proxy | 1940 | 30.08 [28.81, 31.38] | 50.50 [49.50, 51.57] | - | 28.23 / 53.09 |
| E2 long-proxy | 1000 | 32.24 [30.98, 33.46] | 57.88 [57.00, 58.78] | - | 30.31 / 57.47 |
| E2-synth (synthetic) | 300 | 31.97 [30.70, 33.17] | 59.81 [58.91, 60.72] | - | 29.88 / 59.07 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 32.52 [29.63, 35.19] | 59.36 [57.39, 61.23] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 30.04 [28.13, 31.84] | 58.81 [57.17, 60.30] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 33.14 [31.00, 35.15] | 61.27 [59.92, 62.69] | - | - |
| E3 books-proxy | 1000 | 14.36 [13.51, 15.24] | 37.00 [36.09, 37.99] | - | 13.64 / 37.92 |

Source: `s1_sin_l4/seg_tuned/eval.json`.


Length buckets (E1+E2+E3 pooled):

| source words | n | BLEU [95% CI] | chrF [95% CI] |
|---|---|---|---|
| <=10 | 1152 | 25.98 [24.11, 27.95] | 43.53 [41.97, 45.02] |
| 11-20 | 685 | 23.42 [21.65, 25.47] | 44.61 [43.33, 46.08] |
| 21-40 | 1230 | 26.15 [24.95, 27.36] | 51.02 [50.08, 51.95] |
| 41-80 | 795 | 30.10 [28.94, 31.37] | 56.63 [55.72, 57.58] |
| >80 | 78 | 30.52 [26.14, 34.32] | 56.11 [53.23, 59.24] |

Source: `s1_sin_l4/seg_tuned/eval.json` (`length_buckets_e1_e2_e3`).


### S2 RoPE (4,107 steps)

alpha 1.2, beam 5, segmentation T=64; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 29.33 [24.24, 34.50] | 47.25 [43.82, 50.52] | - | 26.81 / 51.63 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 38.98 [35.62, 42.17] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 34.38 [24.52, 44.29] | 60.38 [54.18, 66.48] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 30.27 [22.81, 37.64] | 46.19 [40.11, 52.56] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 19.60 [15.44, 23.72] | 41.75 [38.06, 45.35] | - | - |
| E1 seen-proxy | 1940 | 31.64 [30.36, 32.93] | 51.42 [50.43, 52.51] | - | 29.80 / 54.12 |
| E2 long-proxy | 1000 | 33.24 [31.96, 34.52] | 58.67 [57.81, 59.60] | - | 31.36 / 58.35 |
| E2-synth (synthetic) | 300 | 32.89 [31.63, 34.10] | 60.55 [59.64, 61.46] | - | 30.80 / 59.86 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 33.37 [30.85, 35.76] | 60.15 [58.22, 61.98] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 31.49 [29.42, 33.50] | 59.65 [58.09, 61.21] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 33.67 [31.39, 35.75] | 61.85 [60.47, 63.13] | - | - |
| E3 books-proxy | 1000 | 15.38 [14.51, 16.23] | 38.03 [37.07, 38.97] | - | 14.63 / 39.13 |

Source: `s2_rope_l4/seg_tuned/eval.json`.


Length buckets (E1+E2+E3 pooled):

| source words | n | BLEU [95% CI] | chrF [95% CI] |
|---|---|---|---|
| <=10 | 1152 | 27.38 [25.43, 29.35] | 44.47 [42.89, 46.01] |
| 11-20 | 685 | 24.68 [22.76, 26.75] | 45.54 [44.14, 47.00] |
| 21-40 | 1230 | 27.22 [26.06, 28.35] | 51.88 [50.95, 52.80] |
| 41-80 | 795 | 31.24 [30.03, 32.43] | 57.54 [56.60, 58.46] |
| >80 | 78 | 32.51 [27.94, 36.09] | 57.30 [54.52, 60.49] |

Source: `s2_rope_l4/seg_tuned/eval.json` (`length_buckets_e1_e2_e3`).


### S3 RoPE + concat augmentation (4,107 steps)

alpha 1.2, beam 5, segmentation T=64; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 29.54 [24.35, 34.62] | 47.83 [44.44, 51.41] | - | 27.37 / 51.72 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 39.47 [35.83, 42.89] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 34.55 [24.73, 44.82] | 60.37 [54.12, 66.58] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 29.20 [22.05, 36.41] | 46.78 [40.48, 53.48] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 20.81 [16.09, 25.04] | 42.62 [38.57, 46.58] | - | - |
| E1 seen-proxy | 1940 | 31.66 [30.34, 32.94] | 51.51 [50.48, 52.60] | - | 29.68 / 54.04 |
| E2 long-proxy | 1000 | 33.26 [31.99, 34.53] | 58.72 [57.82, 59.64] | - | 31.38 / 58.36 |
| E2-synth (synthetic) | 300 | 33.15 [31.80, 34.31] | 60.66 [59.75, 61.60] | - | 31.01 / 59.94 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 33.91 [31.18, 36.48] | 60.40 [58.54, 62.28] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 30.98 [28.93, 32.93] | 59.31 [57.82, 60.77] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 34.29 [32.20, 36.13] | 62.28 [60.95, 63.60] | - | - |
| E3 books-proxy | 1000 | 14.98 [14.04, 15.82] | 37.78 [36.82, 38.76] | - | 14.21 / 38.82 |

Source: `s3_rope_concat_l4/seg_tuned/eval.json`.


Length buckets (E1+E2+E3 pooled):

| source words | n | BLEU [95% CI] | chrF [95% CI] |
|---|---|---|---|
| <=10 | 1152 | 26.99 [25.12, 28.90] | 44.54 [42.92, 46.04] |
| 11-20 | 685 | 24.67 [22.76, 26.81] | 45.73 [44.36, 47.15] |
| 21-40 | 1230 | 27.13 [25.94, 28.30] | 51.70 [50.77, 52.62] |
| 41-80 | 795 | 31.13 [29.92, 32.38] | 57.54 [56.57, 58.48] |
| >80 | 78 | 32.59 [28.03, 35.99] | 57.36 [54.53, 60.59] |

Source: `s3_rope_concat_l4/seg_tuned/eval.json` (`length_buckets_e1_e2_e3`).


## Segmentation off vs tuned

### main (deep-enc/shallow-dec, 24,645 steps)

| set | chrF seg OFF | chrF T=192 | delta | BLEU seg OFF | BLEU T=192 | delta |
|---|---|---|---|---|---|---|
| official dev | 50.57 | 50.57 | +0.00 | 32.77 | 32.77 | +0.00 |
| E1 seen-proxy | 54.98 | 54.98 | +0.00 | 35.95 | 35.98 | +0.03 |
| E2 long-proxy | 61.78 | 61.78 | +0.01 | 37.96 | 38.02 | +0.06 |
| E2-synth (synthetic) | 63.24 | 63.31 | +0.07 | 37.61 | 37.73 | +0.12 |
| E3 books-proxy | 42.31 | 42.31 | +0.00 | 19.45 | 19.45 | +0.00 |

Source: `main/seg_off/eval.json`, `main/seg_tuned/eval.json` (differences are tuned minus off, computed from the two JSONs).


### S1 sinusoidal (4,107 steps)

| set | chrF seg OFF | chrF T=64 | delta | BLEU seg OFF | BLEU T=64 | delta |
|---|---|---|---|---|---|---|
| official dev | 46.35 | 46.35 | -0.01 | 27.13 | 27.19 | +0.06 |
| E1 seen-proxy | 50.46 | 50.50 | +0.05 | 29.84 | 30.08 | +0.25 |
| E2 long-proxy | 57.73 | 57.88 | +0.15 | 32.05 | 32.24 | +0.19 |
| E2-synth (synthetic) | 56.04 | 59.81 | +3.77 | 27.63 | 31.97 | +4.34 |
| E3 books-proxy | 36.99 | 37.00 | +0.01 | 14.26 | 14.36 | +0.10 |

Source: `s1_sin_l4/seg_off/eval.json`, `s1_sin_l4/seg_tuned/eval.json` (differences are tuned minus off, computed from the two JSONs).


### S2 RoPE (4,107 steps)

| set | chrF seg OFF | chrF T=64 | delta | BLEU seg OFF | BLEU T=64 | delta |
|---|---|---|---|---|---|---|
| official dev | 47.24 | 47.25 | +0.01 | 29.31 | 29.33 | +0.02 |
| E1 seen-proxy | 51.37 | 51.42 | +0.04 | 31.38 | 31.64 | +0.26 |
| E2 long-proxy | 58.62 | 58.67 | +0.06 | 33.21 | 33.24 | +0.03 |
| E2-synth (synthetic) | 59.15 | 60.55 | +1.40 | 31.76 | 32.89 | +1.13 |
| E3 books-proxy | 38.03 | 38.03 | -0.01 | 15.40 | 15.38 | -0.02 |

Source: `s2_rope_l4/seg_off/eval.json`, `s2_rope_l4/seg_tuned/eval.json` (differences are tuned minus off, computed from the two JSONs).


### S3 RoPE + concat augmentation (4,107 steps)

| set | chrF seg OFF | chrF T=64 | delta | BLEU seg OFF | BLEU T=64 | delta |
|---|---|---|---|---|---|---|
| official dev | 47.84 | 47.83 | -0.01 | 29.58 | 29.54 | -0.04 |
| E1 seen-proxy | 51.49 | 51.51 | +0.02 | 31.53 | 31.66 | +0.13 |
| E2 long-proxy | 58.68 | 58.72 | +0.05 | 33.22 | 33.26 | +0.03 |
| E2-synth (synthetic) | 60.24 | 60.66 | +0.42 | 32.52 | 33.15 | +0.63 |
| E3 books-proxy | 37.78 | 37.78 | +0.00 | 14.98 | 14.98 | +0.00 |

Source: `s3_rope_concat_l4/seg_off/eval.json`, `s3_rope_concat_l4/seg_tuned/eval.json` (differences are tuned minus off, computed from the two JSONs).


## Per-model results with segmentation OFF (primary decoding for H1/H2)

### main (deep-enc/shallow-dec, 24,645 steps)

alpha 1.2, beam 5, segmentation OFF; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 32.77 [26.98, 38.05] | 50.57 [46.86, 54.39] | - | 30.91 / 54.45 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 42.29 [38.72, 45.82] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 38.44 [28.07, 48.97] | 63.16 [56.96, 69.79] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 32.43 [24.56, 40.39] | 50.08 [43.30, 57.61] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 21.99 [17.98, 26.06] | 44.76 [40.74, 48.39] | - | - |
| E1 seen-proxy | 1940 | 35.95 [34.64, 37.29] | 54.98 [53.93, 56.11] | - | 33.82 / 57.44 |
| E2 long-proxy | 1000 | 37.96 [36.57, 39.31] | 61.78 [60.81, 62.70] | - | 36.06 / 61.40 |
| E2-synth (synthetic) | 300 | 37.61 [36.24, 38.94] | 63.24 [62.24, 64.21] | - | 35.68 / 62.58 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 37.78 [35.01, 40.40] | 62.85 [60.93, 64.82] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 36.28 [33.90, 38.56] | 62.17 [60.44, 63.80] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 38.63 [36.12, 40.89] | 64.70 [63.23, 66.14] | - | - |
| E3 books-proxy | 1000 | 19.45 [18.41, 20.52] | 42.31 [41.29, 43.27] | - | 18.77 / 42.83 |

Source: `main/seg_off/eval.json`.


### S1 sinusoidal (4,107 steps)

alpha 1.2, beam 4, segmentation OFF; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 27.13 [21.90, 32.15] | 46.35 [42.88, 49.68] | - | 25.16 / 50.40 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 37.49 [33.98, 40.56] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 31.74 [22.25, 41.52] | 59.19 [53.16, 65.06] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 27.28 [20.10, 34.26] | 45.79 [39.47, 52.24] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 17.81 [14.00, 21.53] | 40.50 [36.75, 44.18] | - | - |
| E1 seen-proxy | 1940 | 29.84 [28.62, 31.16] | 50.46 [49.43, 51.53] | - | 28.01 / 52.78 |
| E2 long-proxy | 1000 | 32.05 [30.79, 33.17] | 57.73 [56.84, 58.60] | - | 30.20 / 57.17 |
| E2-synth (synthetic) | 300 | 27.63 [26.44, 28.86] | 56.04 [55.08, 57.02] | - | 26.05 / 54.83 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 31.63 [29.04, 34.55] | 58.33 [56.43, 60.13] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 26.49 [24.53, 28.58] | 55.28 [53.56, 56.97] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 26.06 [24.19, 27.82] | 54.52 [52.97, 56.06] | - | - |
| E3 books-proxy | 1000 | 14.26 [13.40, 15.16] | 36.99 [36.08, 37.97] | - | 13.63 / 37.85 |

Source: `s1_sin_l4/seg_off/eval.json`.


### S2 RoPE (4,107 steps)

alpha 1.2, beam 5, segmentation OFF; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 29.31 [24.20, 34.50] | 47.24 [43.81, 50.52] | - | 26.79 / 51.61 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 38.97 [35.61, 42.16] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 34.33 [24.48, 44.20] | 60.32 [53.99, 66.54] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 30.27 [22.81, 37.64] | 46.19 [40.11, 52.56] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 19.60 [15.44, 23.72] | 41.75 [38.06, 45.35] | - | - |
| E1 seen-proxy | 1940 | 31.38 [30.12, 32.69] | 51.37 [50.38, 52.46] | - | 29.54 / 53.95 |
| E2 long-proxy | 1000 | 33.21 [31.90, 34.38] | 58.62 [57.73, 59.52] | - | 31.36 / 58.25 |
| E2-synth (synthetic) | 300 | 31.76 [30.45, 33.02] | 59.15 [58.11, 60.15] | - | 30.03 / 58.50 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 32.34 [29.83, 35.13] | 58.36 [56.22, 60.50] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 30.37 [28.19, 32.23] | 58.72 [57.16, 60.35] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 32.55 [30.33, 34.70] | 60.37 [58.64, 61.96] | - | - |
| E3 books-proxy | 1000 | 15.40 [14.54, 16.26] | 38.03 [37.05, 38.99] | - | 14.78 / 39.13 |

Source: `s2_rope_l4/seg_off/eval.json`.


### S3 RoPE + concat augmentation (4,107 steps)

alpha 1.2, beam 5, segmentation OFF; checkpoint `final`; sacreBLEU BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | COMET-22 [95% CI] | sacreBLEU BLEU / chrF |
|---|---|---|---|---|---|
| official dev | 150 | 29.58 [24.35, 34.62] | 47.84 [44.44, 51.42] | - | 27.45 / 51.76 |
| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | 150 | 39.49 [35.83, 42.90] | - | - | - |
| &nbsp;&nbsp;dev: long | 30 | 34.65 [24.79, 45.01] | 60.40 [54.14, 66.61] | - | - |
| &nbsp;&nbsp;dev: seen | 60 | 29.20 [22.05, 36.41] | 46.78 [40.48, 53.48] | - | - |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 20.81 [16.09, 25.04] | 42.62 [38.57, 46.58] | - | - |
| E1 seen-proxy | 1940 | 31.53 [30.26, 32.83] | 51.49 [50.46, 52.58] | - | 29.58 / 53.97 |
| E2 long-proxy | 1000 | 33.22 [31.96, 34.47] | 58.68 [57.76, 59.57] | - | 31.36 / 58.30 |
| E2-synth (synthetic) | 300 | 32.52 [31.29, 33.69] | 60.24 [59.35, 61.17] | - | 30.63 / 59.49 |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 33.34 [30.77, 35.90] | 59.94 [58.06, 61.81] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 30.58 [28.62, 32.38] | 59.13 [57.62, 60.60] | - | - |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 33.49 [31.34, 35.39] | 61.66 [60.17, 63.05] | - | - |
| E3 books-proxy | 1000 | 14.98 [14.04, 15.85] | 37.78 [36.82, 38.76] | - | 14.30 / 38.84 |

Source: `s3_rope_concat_l4/seg_off/eval.json`.


## Pre-registered paired tests (PREREG §3)

**H1 (PRIMARY, segmentation off; delta = s2_rope_l4 - s1_sin_l4, chrF):** RoPE improves chrF on long inputs -> E2 +0.89 [+0.66, +1.14] p < 0.001; E2-synth +3.11 [+2.40, +3.85] p < 0.001 -> **SUPPORTED**

Decision-rule components (checked = met):

- [x] E2 chrF (n=1000): delta +0.89 [+0.66, +1.14], p < 0.001
- [x] E2-synth chrF, pooled over 3 buckets (n=300): delta +3.11 [+2.40, +3.85], p < 0.001

Also reported, not decisive:

  - dev seen chrF: delta +0.40 [-1.45, +2.25], p 0.341
  - dev long chrF: delta +1.12 [-0.09, +2.37], p 0.044
  - E2-synth e2synth_400_600 chrF: delta +0.03 [-1.31, +1.20], p 0.483
  - E2-synth e2synth_600_800 chrF: delta +3.45 [+2.33, +4.58], p < 0.001
  - E2-synth e2synth_800_900 chrF: delta +5.85 [+4.66, +7.01], p < 0.001
  - >80-word bucket E1+E2+E3 (n=78) chrF: delta +2.45 [+1.52, +3.47], p < 0.001
  - >80-word bucket E1+E2+E3 (n=78) BLEU: delta +3.62 [+1.59, +5.12], p < 0.001
  - E2 BLEU: delta +1.15 [+0.74, +1.65], p < 0.001
  - E2-synth BLEU: delta +4.13 [+3.35, +4.94], p < 0.001
  - selection objective: delta +1.12 [+0.87, +1.34], p < 0.001

Source: `compare/H1_s2_rope_l4_vs_s1_sin_l4_seg_off.json`, `compare/extras.json`.

**H1 (secondary, tuned; delta = s2_rope_l4 - s1_sin_l4, chrF):** RoPE improves chrF on long inputs -> E2 +0.80 [+0.57, +1.04] p < 0.001; E2-synth +0.74 [+0.40, +1.05] p < 0.001 -> **SUPPORTED**

Decision-rule components (checked = met):

- [x] E2 chrF (n=1000): delta +0.80 [+0.57, +1.04], p < 0.001
- [x] E2-synth chrF, pooled over 3 buckets (n=300): delta +0.74 [+0.40, +1.05], p < 0.001

Also reported, not decisive:

  - dev seen chrF: delta +0.40 [-1.45, +2.25], p 0.341
  - dev long chrF: delta +1.21 [-0.06, +2.50], p 0.032
  - E2-synth e2synth_400_600 chrF: delta +0.79 [+0.17, +1.40], p 0.006
  - E2-synth e2synth_600_800 chrF: delta +0.84 [+0.33, +1.44], p 0.001
  - E2-synth e2synth_800_900 chrF: delta +0.59 [+0.15, +1.01], p 0.008
  - >80-word bucket E1+E2+E3 (n=78) chrF: delta +1.19 [+0.60, +1.84], p < 0.001
  - >80-word bucket E1+E2+E3 (n=78) BLEU: delta +1.99 [+0.65, +3.12], p < 0.001
  - E2 BLEU: delta +0.99 [+0.64, +1.41], p < 0.001
  - E2-synth BLEU: delta +0.92 [+0.37, +1.51], p < 0.001
  - selection objective: delta +1.06 [+0.82, +1.29], p < 0.001

Source: `compare/H1_s2_rope_l4_vs_s1_sin_l4_seg_tuned.json`, `compare/extras.json`.

**H2 (PRIMARY, segmentation off; delta = s3_rope_concat_l4 - s2_rope_l4, chrF):** concat augmentation helps long inputs without hurting seen inputs by more than 0.5 chrF -> E2 +0.06 [-0.19, +0.29] p 0.313; E2-synth +1.09 [+0.59, +1.68] p < 0.001 -> **NOT SUPPORTED**

Decision-rule components (checked = met):

- [ ] E2 chrF (n=1000): delta +0.06 [-0.19, +0.29], p 0.313
- [x] E2-synth chrF, pooled over 3 buckets (n=300): delta +1.09 [+0.59, +1.68], p < 0.001
- [x] E1 non-inferiority (CI lower bound > -0.5): E1 chrF (n=1940): delta +0.12 [-0.22, +0.44], p 0.277; lower bound -0.22

Also reported, not decisive:

  - dev seen chrF: delta +0.59 [-1.56, +2.51], p 0.271
  - dev long chrF: delta +0.08 [-0.90, +1.16], p 0.447
  - E2-synth e2synth_400_600 chrF: delta +1.59 [+0.37, +2.87], p < 0.001
  - E2-synth e2synth_600_800 chrF: delta +0.41 [-0.13, +1.03], p 0.071
  - E2-synth e2synth_800_900 chrF: delta +1.29 [+0.35, +2.41], p 0.004
  - >80-word bucket E1+E2+E3 (n=78) chrF: delta +0.46 [-0.16, +1.09], p 0.069
  - >80-word bucket E1+E2+E3 (n=78) BLEU: delta +0.85 [-0.15, +1.84], p 0.057
  - E2 BLEU: delta +0.02 [-0.33, +0.41], p 0.406
  - E2-synth BLEU: delta +0.76 [+0.15, +1.30], p 0.004
  - selection objective: delta +0.11 [-0.16, +0.35], p 0.221

Source: `compare/H2_s3_rope_concat_l4_vs_s2_rope_l4_seg_off.json`, `compare/extras.json`.

**H2 (secondary, tuned; delta = s3_rope_concat_l4 - s2_rope_l4, chrF):** concat augmentation helps long inputs without hurting seen inputs by more than 0.5 chrF -> E2 +0.05 [-0.19, +0.28] p 0.335; E2-synth +0.11 [-0.15, +0.39] p 0.210 -> **NOT SUPPORTED**

Decision-rule components (checked = met):

- [ ] E2 chrF (n=1000): delta +0.05 [-0.19, +0.28], p 0.335
- [ ] E2-synth chrF, pooled over 3 buckets (n=300): delta +0.11 [-0.15, +0.39], p 0.210
- [x] E1 non-inferiority (CI lower bound > -0.5): E1 chrF (n=1940): delta +0.09 [-0.24, +0.42], p 0.321; lower bound -0.24

Also reported, not decisive:

  - dev seen chrF: delta +0.59 [-1.56, +2.51], p 0.271
  - dev long chrF: delta -0.01 [-0.99, +1.13], p 0.516
  - E2-synth e2synth_400_600 chrF: delta +0.25 [-0.32, +0.79], p 0.173
  - E2-synth e2synth_600_800 chrF: delta -0.34 [-0.78, +0.11], p 0.942
  - E2-synth e2synth_800_900 chrF: delta +0.43 [-0.04, +0.86], p 0.035
  - >80-word bucket E1+E2+E3 (n=78) chrF: delta +0.06 [-0.45, +0.64], p 0.416
  - >80-word bucket E1+E2+E3 (n=78) BLEU: delta +0.08 [-0.76, +0.86], p 0.433
  - E2 BLEU: delta +0.02 [-0.31, +0.39], p 0.470
  - E2-synth BLEU: delta +0.26 [-0.25, +0.66], p 0.191
  - selection objective: delta +0.04 [-0.19, +0.29], p 0.360

Source: `compare/H2_s3_rope_concat_l4_vs_s2_rope_l4_seg_tuned.json`, `compare/extras.json`.

## Analysis

### Gap decomposition (OLS over E1 union E3)

| model | variant | E1-E3 chrF gap | length_dev | repetition | rarity | dialogue | residual domain | R^2 | n |
|---|---|---|---|---|---|---|---|---|---|
| main | seg_off | 12.67 | -0.50 | +0.01 | +0.08 | +0.15 | +12.93 | 0.188 | 2940 |
| main | seg_tuned | 12.67 | -0.49 | +0.05 | +0.08 | +0.16 | +12.88 | 0.178 | 2940 |
| s1_sin_l4 | seg_off | 13.47 | -0.44 | +0.03 | +0.05 | +0.31 | +13.53 | 0.190 | 2940 |
| s1_sin_l4 | seg_tuned | 13.51 | -0.42 | +0.12 | +0.04 | +0.31 | +13.46 | 0.184 | 2940 |
| s2_rope_l4 | seg_off | 13.34 | -0.29 | +0.03 | +0.07 | +0.28 | +13.25 | 0.183 | 2940 |
| s2_rope_l4 | seg_tuned | 13.39 | -0.29 | +0.10 | +0.06 | +0.28 | +13.24 | 0.179 | 2940 |
| s3_rope_concat_l4 | seg_off | 13.70 | -0.63 | +0.02 | +0.05 | +0.29 | +13.97 | 0.193 | 2940 |
| s3_rope_concat_l4 | seg_tuned | 13.73 | -0.61 | +0.11 | +0.05 | +0.29 | +13.88 | 0.186 | 2940 |

Contributions are chrF points of the E1-E3 gap (coefficient x group-mean difference; residual domain = -coef(domain_e3)). Source: `<run>/<variant>/analysis.json` (`gap_decomposition`, OLS with HC3 robust SEs).

### Reference-normalisation artifact

| model | variant | dev unseen chrF (orig -> ref-normalised, delta) | E3 chrF (orig -> ref-normalised, delta) |
|---|---|---|---|
| main | seg_off | 44.76 -> 45.01 (+0.25) | 42.31 -> 42.53 (+0.22) |
| main | seg_tuned | 44.76 -> 45.01 (+0.25) | 42.31 -> 42.53 (+0.22) |
| s1_sin_l4 | seg_off | 40.50 -> 40.75 (+0.25) | 36.99 -> 37.22 (+0.23) |
| s1_sin_l4 | seg_tuned | 40.50 -> 40.75 (+0.25) | 37.00 -> 37.23 (+0.23) |
| s2_rope_l4 | seg_off | 41.75 -> 41.99 (+0.24) | 38.03 -> 38.26 (+0.23) |
| s2_rope_l4 | seg_tuned | 41.75 -> 41.99 (+0.24) | 38.03 -> 38.25 (+0.23) |
| s3_rope_concat_l4 | seg_off | 42.62 -> 42.89 (+0.27) | 37.78 -> 38.04 (+0.25) |
| s3_rope_concat_l4 | seg_tuned | 42.62 -> 42.89 (+0.27) | 37.78 -> 38.04 (+0.25) |

Source: `<run>/<variant>/analysis.json` (`metric_artifact_share`).

### Failure-mode rates per slice (tuned config)

| model | group | n | repetition | truncation | untranslated copy | over-long | mean hyp/ref ratio |
|---|---|---|---|---|---|---|---|
| main | dev:seen | 60 | 0.000 | 0.033 | 0.000 | 0.083 | 1.059 |
| main | dev:long | 30 | 0.000 | 0.000 | 0.000 | 0.067 | 1.076 |
| main | dev:unseen_domain | 60 | 0.000 | 0.000 | 0.000 | 0.017 | 0.991 |
| main | e1 | 1940 | 0.005 | 0.022 | 0.000 | 0.040 | 1.013 |
| main | e2 | 1000 | 0.030 | 0.007 | 0.000 | 0.037 | 1.029 |
| main | e2synth | 300 | 0.073 | 0.000 | 0.000 | 0.000 | 1.005 |
| main | e3 | 1000 | 0.003 | 0.012 | 0.000 | 0.023 | 0.983 |
| s1_sin_l4 | dev:seen | 60 | 0.000 | 0.033 | 0.000 | 0.067 | 1.048 |
| s1_sin_l4 | dev:long | 30 | 0.000 | 0.000 | 0.000 | 0.067 | 1.099 |
| s1_sin_l4 | dev:unseen_domain | 60 | 0.000 | 0.017 | 0.000 | 0.017 | 0.981 |
| s1_sin_l4 | e1 | 1940 | 0.014 | 0.024 | 0.000 | 0.045 | 1.013 |
| s1_sin_l4 | e2 | 1000 | 0.035 | 0.005 | 0.000 | 0.036 | 1.044 |
| s1_sin_l4 | e2synth | 300 | 0.080 | 0.000 | 0.000 | 0.003 | 1.015 |
| s1_sin_l4 | e3 | 1000 | 0.002 | 0.023 | 0.000 | 0.029 | 0.985 |
| s2_rope_l4 | dev:seen | 60 | 0.000 | 0.033 | 0.000 | 0.083 | 1.079 |
| s2_rope_l4 | dev:long | 30 | 0.000 | 0.000 | 0.000 | 0.067 | 1.101 |
| s2_rope_l4 | dev:unseen_domain | 60 | 0.000 | 0.000 | 0.000 | 0.033 | 1.023 |
| s2_rope_l4 | e1 | 1940 | 0.014 | 0.024 | 0.000 | 0.042 | 1.013 |
| s2_rope_l4 | e2 | 1000 | 0.042 | 0.005 | 0.000 | 0.037 | 1.049 |
| s2_rope_l4 | e2synth | 300 | 0.113 | 0.000 | 0.000 | 0.003 | 1.023 |
| s2_rope_l4 | e3 | 1000 | 0.003 | 0.016 | 0.000 | 0.032 | 0.994 |
| s3_rope_concat_l4 | dev:seen | 60 | 0.000 | 0.033 | 0.000 | 0.067 | 1.062 |
| s3_rope_concat_l4 | dev:long | 30 | 0.000 | 0.000 | 0.000 | 0.067 | 1.100 |
| s3_rope_concat_l4 | dev:unseen_domain | 60 | 0.000 | 0.000 | 0.000 | 0.033 | 1.010 |
| s3_rope_concat_l4 | e1 | 1940 | 0.014 | 0.021 | 0.000 | 0.046 | 1.019 |
| s3_rope_concat_l4 | e2 | 1000 | 0.037 | 0.006 | 0.000 | 0.035 | 1.048 |
| s3_rope_concat_l4 | e2synth | 300 | 0.083 | 0.000 | 0.000 | 0.003 | 1.018 |
| s3_rope_concat_l4 | e3 | 1000 | 0.004 | 0.013 | 0.000 | 0.027 | 0.988 |

Shares of sentences (definitions in `<run>/seg_tuned/diagnostics.json`). Source: `<run>/seg_tuned/diagnostics.json` (`failure_modes`).

### Chrf by source-rarity bucket (tuned config)

| model | bucket 1 | bucket 2 | bucket 3 | bucket 4 | bucket 5 |
|---|---|---|---|---|---|
| main | 55.19 | 56.05 | 53.45 | 52.31 | 50.45 |
| s1_sin_l4 | 50.48 | 51.92 | 49.17 | 47.85 | 45.32 |
| s2_rope_l4 | 51.42 | 52.56 | 50.29 | 48.83 | 46.20 |
| s3_rope_concat_l4 | 51.26 | 52.60 | 50.20 | 49.07 | 46.15 |

Mean sentence chrF per source-rarity quintile over E1+E2+E3 (bucket 1 = most common sources, 5 = rarest; n per bucket 788, 788, 788, 788, 788). Source: `<run>/seg_tuned/diagnostics.json` (`rarity_buckets_e1_e2_e3`).

Length buckets are in each model's table above; failure examples are in `EXAMPLES.md`.

## Sanity checks

- Pull records verified for all 4 runs: True.
- Fast-bootstrap point estimates vs official CLI numbers: 80 comparisons, max abs diff 0.0.
- Dev prediction file re-scored through the wrapper equals eval.json for 8/8 (run, variant) pairs.
- Selection objective re-scored locally (E1 unsegmented + E2 at tuned T, as nmt.tune scores it) vs `selection.json`, max abs diff 0.00e+00.
- NOT PRESENT: the uploaded runs/<run>/ contain no dev scores (selection.json and tuning/ use E1+E2 only; dev is decoded after selection), so the dev re-score is checked against the in-process scorer and the CLI parity record, not a Colab-side figure.

Source: `sanity.json`.

## Provenance

- Code: commit `695a8d7d47bab33f30fba05252af31b301f754b2` (HEAD when the artifacts were generated), eval tag of the Colab decodes `v0.2.4-colab` (adb3c8c781dc70e204e0f48d1a48123b7bcd261f).
- Bootstrap: 1,000 resamples, seed 1234, 95% percentile CIs; paired tests share resample indices.
- HF repo `OWNER/fr-en-transformer-eval` (private), pinned revisions (each `pull_record.json` has `verified: true`):
  - `main`: `c3d8598252853fcd7df1ef4a00e8b0382b8f4351`
  - `s1_sin_l4`: `041d49269f61d0cbd9c0380a4f0a0a31f7599547`
  - `s2_rope_l4`: `bdd850a6d384e088dd1eb52d67ad767149dee3af`
  - `s3_rope_concat_l4`: `ac92b8da971fdf64974089f40e8f4ddbf9d9638b`

Files (sha256):

| file | sha256 |
|---|---|
| `main/pull_record.json` | `a78f373ab8cb0bb7c80bb98672e38c79b5e6c6955eb11b318c0bfca631d45279` |
| `main/index.json` | `3a22a4f22af3f5ee0741b44eed6eea4cdafec4ce50a96150f4b52c1a0c381a3f` |
| `main/selection.json` | `132ab16997ea349f7bb62b11a5568ec4094bc813ef3e41b5028ee751df8619c8` |
| `main/seg_off/eval.json` | `e01abda0cb5903fe0585ddc4e9b84b1b80c75663d867891e046a38ede8a694f0` |
| `main/seg_off/analysis.json` | `75538cc2045c23dc485916915eb1edbb2ab07ef3f68689f753d7d348f5516801` |
| `main/seg_off/diagnostics.json` | `18f8697381a16eb8eb845c3ec67c5d6c9fa3d300dd2a105d04c6d05c3ee5268d` |
| `main/seg_off/examples.json` | `0285eb545a81c81d20f8b48bde3689ef69fb725ef5d148997ef0376819224205` |
| `main/seg_tuned/eval.json` | `f3ce82a632969f8e229f842577b670b46d3b3d73a6348f82093abc6eb176673c` |
| `main/seg_tuned/analysis.json` | `f6c0821d7b576cfd43692fe25e3ff29b3334f1bde9622e05c2c198865da62752` |
| `main/seg_tuned/diagnostics.json` | `b6648c4553d06f04018c84f9f735676fd2bf268683fc79b0fd2f0aae821e3a7f` |
| `main/seg_tuned/examples.json` | `0285eb545a81c81d20f8b48bde3689ef69fb725ef5d148997ef0376819224205` |
| `s1_sin_l4/pull_record.json` | `f12f78c20b701d7ab5bdb1c985a8e4db3039f6fde83766f4ab3c4907f7aaeede` |
| `s1_sin_l4/index.json` | `9c19485a1a2d10646b98becff5ca7271c4c55dc22532eeea2d97ea045516f296` |
| `s1_sin_l4/selection.json` | `721ce49134730eb79bbfdd4b504c11bb58284b4e352c667b3ad4bde240d6b065` |
| `s1_sin_l4/seg_off/eval.json` | `46ee3e99bbd5e64e2d08b5b8b5d4e0d52fec2ff748f487d780515e598a6cadb4` |
| `s1_sin_l4/seg_off/analysis.json` | `298c770fd9330f5da11c1268ab0b74207d7f23d319d999e3de1bf63adb344b92` |
| `s1_sin_l4/seg_off/diagnostics.json` | `fba6991a689a6fc08b67bee7707da201c325b305f6dc22e2e82f4297b208ed55` |
| `s1_sin_l4/seg_off/examples.json` | `f579fa57872f3a183babebe6e8b74e25c9923e1ef7e58679f7067028d96fc187` |
| `s1_sin_l4/seg_tuned/eval.json` | `1dc9b9bdcb93caae6b196ece560f68134f2af06c359b3e9538c10a21129d49eb` |
| `s1_sin_l4/seg_tuned/analysis.json` | `4f36bd3ae29b9d90df4ca08c932d240db389406643e3613f0b5cecba08d6bdc0` |
| `s1_sin_l4/seg_tuned/diagnostics.json` | `d009a1e47a0fc1983992da0be11b15c612fa3097063e9d9cfd261361d73772ed` |
| `s1_sin_l4/seg_tuned/examples.json` | `762b823e9bd6fa6e9ff72f211c5a4a928d9913d59fcc68b8b57be6f241e3be95` |
| `s2_rope_l4/pull_record.json` | `582f3b0330f2df3efe778ac75173824f0b5fd49ec75231a3789dcae6d9186a7a` |
| `s2_rope_l4/index.json` | `2359a3950d197dc380ef05eaae3ff9471e6558a4c7bb686d5c654c74a6829342` |
| `s2_rope_l4/selection.json` | `841752e2dd9142249aad5035386066c77c2c029b1dc6f3dad6e361c64a376c05` |
| `s2_rope_l4/seg_off/eval.json` | `c1946276dfb36fdd6fce49c80b28a5c2fb96a68beea542004c4442e1fb726323` |
| `s2_rope_l4/seg_off/analysis.json` | `63b7ac8253d49e333a32bcba5ee22448de2442d3a910108a58b98acd7725f7d5` |
| `s2_rope_l4/seg_off/diagnostics.json` | `9946265ea1f84d377c67ca60a3f0526994644128e6981ff4d15e135c42dcee71` |
| `s2_rope_l4/seg_off/examples.json` | `315e625665e09ca3ccde0eabd17d9f27f2079ae861b2c3d2a6b6d2ccdd69f22e` |
| `s2_rope_l4/seg_tuned/eval.json` | `5052056e9f278208305a5110459a327d64d4878a331e214d4d3b34f785c969e2` |
| `s2_rope_l4/seg_tuned/analysis.json` | `5b50d02cc8fc3d5d90fd2c4aba600118a1e9ff3abac6fd764d61362fc7e20fae` |
| `s2_rope_l4/seg_tuned/diagnostics.json` | `7f079f52401fa47a28dcd4ee36e2a57d83d6da99fbd11e60bc5ea9d6bc2705f3` |
| `s2_rope_l4/seg_tuned/examples.json` | `bd54d57e37a6748bf63cead6ad350630c4d249f39f352685eb66fbf0243daa7f` |
| `s3_rope_concat_l4/pull_record.json` | `f58ae4ad1023c2ce37269fdd7846a878490e9be4a9771e2e67d110dcc84426ef` |
| `s3_rope_concat_l4/index.json` | `3c87f2c17081450cc0eef435cf06308a27f23c92c8e9bea0101845a2b3b4fb44` |
| `s3_rope_concat_l4/selection.json` | `8a1e88f315e18a269ba81478964346f4c5a48b45566031ca82c4e49127608e01` |
| `s3_rope_concat_l4/seg_off/eval.json` | `2e8fce07c4d28eb7da63f1686e08b7328fcc01095759321aab7d40770cc5bb42` |
| `s3_rope_concat_l4/seg_off/analysis.json` | `6bd9a06822d5c56eeb23cf5e747c74c3292b6e1137f488fb0bea8192e3c58cb8` |
| `s3_rope_concat_l4/seg_off/diagnostics.json` | `78248b26dfba7452cbfd38da80eb5518ba542c690ff428ba0db9d10fa980ff0b` |
| `s3_rope_concat_l4/seg_off/examples.json` | `c852cc8d737a73486a49d0b9ac8d7d433175be4ba7cb5e0cb4f4fb10fa326b18` |
| `s3_rope_concat_l4/seg_tuned/eval.json` | `f3b23e6d1b066b399c3055f0f77d3755fda46d3d6f6d731cc36a716ac18e46c2` |
| `s3_rope_concat_l4/seg_tuned/analysis.json` | `57dab66028596a7c025dcb4df20f64e307f56402abccb8ec46f0b5f91a9935d7` |
| `s3_rope_concat_l4/seg_tuned/diagnostics.json` | `6ac793db852a9bffe8e303a5d1c94a66c39d5eec32702c84b4c576cf508b1bd8` |
| `s3_rope_concat_l4/seg_tuned/examples.json` | `66dfe19d8176c6496223de6551b4f1fc1842d0521168229c74d52e36d2551fe0` |
| `compare/extras.json` | `e6b94d36a9a22f43dab53ab7ce68f7dfba6fe162e29235b43b830166de10177a` |
| `compare/H1_s2_rope_l4_vs_s1_sin_l4_seg_off.json` | `e6aa88d3b56ad62414ab6e5442e7e3c9c99cf57dc08943176dcca7c20d30d375` |
| `compare/H1_s2_rope_l4_vs_s1_sin_l4_seg_tuned.json` | `45522195325693fda07cfe80a67d63b316c77f8f6084f089c3a24b61757d377a` |
| `compare/H2_s3_rope_concat_l4_vs_s2_rope_l4_seg_off.json` | `8ab0e3c8d9c19bd3946c6c6adcbed5f324f6cee8889cf67daf1c0983a94e6d04` |
| `compare/H2_s3_rope_concat_l4_vs_s2_rope_l4_seg_tuned.json` | `f93aa12f8747a74dc04ef99bf5cdcfc1de2129aaad6ce6594b7eeaf7587007d0` |
| `compare/index.json` | `d19f6d1dfda94a277b25aa92a769b71f55cebb8004b1b2b30671666ded842c62` |
| `sanity.json` | `5925bb4b4147b4ba97772426b5d860f214d16d0c6bc4c67300baf2d4093a53ec` |
