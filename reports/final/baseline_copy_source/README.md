# copy-source baseline (floor)

Output = the French source sentence verbatim, for every sentence of official dev, E1, E2,
E2-synth and E3. No model, no decoding, no tuning; scored by the same code as the 4 runs
(`nmt.evaluate.score_split_predictions`, official `score.py` through
`nmt.evaluate.run_official_scorer`, 1,000 resamples, seed 1234). No test-set file is written.

## Scores (official BLEU / chrF, 95% bootstrap CI)

| set / slice | n | BLEU [95% CI] | chrF [95% CI] |
|---|---|---|---|
| dev | 150 | 6.91 [3.53, 10.90] | 24.37 [21.84, 27.14] |
| &nbsp;&nbsp;dev: long | 30 | 9.89 [3.20, 17.38] | 35.87 [30.40, 42.38] |
| &nbsp;&nbsp;dev: seen | 60 | 6.41 [3.49, 9.47] | 25.05 [20.37, 30.86] |
| &nbsp;&nbsp;dev: unseen_domain | 60 | 1.76 [0.45, 3.64] | 17.93 [15.77, 20.36] |
| e1 | 1940 | 6.03 [5.10, 7.12] | 26.56 [25.83, 27.33] |
| e2 | 1000 | 5.19 [4.34, 6.12] | 31.98 [31.52, 32.52] |
| e2synth | 300 | 5.42 [4.46, 6.46] | 34.79 [34.22, 35.44] |
| &nbsp;&nbsp;e2synth: e2synth_400_600 | 100 | 5.31 [3.88, 6.70] | 34.20 [33.21, 35.29] |
| &nbsp;&nbsp;e2synth: e2synth_600_800 | 100 | 5.52 [3.80, 7.73] | 34.61 [33.55, 35.88] |
| &nbsp;&nbsp;e2synth: e2synth_800_900 | 100 | 5.40 [3.95, 7.14] | 35.54 [34.79, 36.32] |
| e3 | 1000 | 1.25 [0.95, 1.54] | 20.47 [20.00, 20.98] |

dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen): 16.10 [13.96, 18.74].
Selection objective (0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1)): 18.8954 (`objective.json`).
Source: `eval.json`, `objective.json`.

## Word-copy heuristic calibration

Heuristic (`scripts.build_final_report.word_copies`): hypothesis words of length >= 4 equal to a source word and absent from the reference. Reference level: share of reference words of length >= 4 that equal a source word (names, numbers, cognates the reference itself keeps). Model = main/seg_tuned. Source: `calibration.json`.

| group | n | model: sentences (words) | baseline ceiling: sentences (words) | reference level: sentences (words) |
|---|---|---|---|---|
| dev | 150 | 0.247 (0.033) | 0.973 (0.879) | 0.427 (0.125) |
| dev:long | 30 | 0.500 (0.027) | 0.967 (0.839) | 0.833 (0.176) |
| dev:seen | 60 | 0.217 (0.055) | 0.950 (0.883) | 0.333 (0.107) |
| dev:unseen_domain | 60 | 0.150 (0.024) | 1.000 (0.944) | 0.317 (0.058) |
| e1 | 1940 | 0.255 (0.041) | 0.968 (0.881) | 0.515 (0.127) |
| e2 | 1000 | 0.503 (0.036) | 0.999 (0.884) | 0.829 (0.126) |
| e2synth | 300 | 0.803 (0.037) | 1.000 (0.886) | 0.977 (0.124) |
| e2synth:e2synth_400_600 | 100 | 0.770 (0.040) | 1.000 (0.885) | 0.930 (0.124) |
| e2synth:e2synth_600_800 | 100 | 0.750 (0.036) | 1.000 (0.886) | 1.000 (0.123) |
| e2synth:e2synth_800_900 | 100 | 0.890 (0.037) | 1.000 (0.886) | 1.000 (0.124) |
| e3 | 1000 | 0.323 (0.124) | 0.989 (0.932) | 0.437 (0.069) |

Each cell is the share of sentences with at least one flagged word, and in parentheses the
share of length >= 4 words flagged (hypothesis words for model and baseline, reference
words for the reference level).

## What the heuristic can and cannot tell

On a pure copy of the source the word-copy heuristic flags 0.839 to 0.944 of the length >= 4 words across groups (the ceiling), while main/seg_tuned has 0.024 to 0.124 flagged, so the word share separates a pure copy from this model in every group. The ceiling is an upper bound, not a detection rate, and the heuristic is not validated against labelled copy errors. The sentence-level rate is not comparable across sets: the model's own sentence rate ranges 0.150 to 0.890 across groups and is highest on the long E2-synth inputs while its word share stays low, so the word share is the figure to read (the sentence rate is consistent with tracking sentence length; that was not tested directly). The flagged words are words the reference does not use, so the heuristic cannot tell an untranslated word from a name spelled differently in the reference or a cognate chosen where the reference paraphrased, and it misses copied words shorter than 4 characters or that also occur in the reference. The reference level (words the reference itself keeps from the source: 0.058 to 0.176 of its length >= 4 words) shows such shared words are common and legitimate (E3, the books proxy, has the highest model share, 0.124, above its reference level of 0.069; it was not examined), but they are excluded from the heuristic by construction, so it is context for the ceiling, not a value to subtract. The older subword-based `untranslated_copy_rate` is a weak detector even on a pure copy (0.100 to 0.517 across groups for the baseline, against 0.000 for the model), so its 0.000 for the model is not evidence of the absence of copying.

## Provenance

- Code: commit `ba77a6dedc42f20f6d940e338e8fd0fade8164e1` (HEAD when generated).
- Regenerate (from a clean tree, PYTHONUTF8=1, in this order):
  - `python -m scripts.copy_source_baseline`
  - `python -m scripts.build_final_report --out-root reports/final --reuse-sanity`
- Input sha256 (LF, no CR in any generated text file):

| file | sha256 |
|---|---|
| `data/dev/inputs.jsonl` | `70a4e42020cd6ad0c882d68a18415e07a1345c287bcbe34e7591e6c3d9cffecc` |
| `data/dev/labels.jsonl` | `3f604c53f2c0c9853dc84a607441ee4f4fe4329d8248132b4f37bd062fbec52e` |
| `data/eval/e1/inputs.jsonl` | `f66cf448b09bacee41a45bb77cd12a61fa5f4d9f821f005cd9da8e35da41c6be` |
| `data/eval/e1/labels.jsonl` | `6187337411098dcb663378867e2b17182f1e0d118aacdc09cbba5f3df1fd0fc9` |
| `data/eval/e2/inputs.jsonl` | `a0ecfe96974c9c01eba2043fb9e3ceb046cfc1da523b10890ec3f46003e1bf27` |
| `data/eval/e2/labels.jsonl` | `d5d122bd57ba9f49e9d65a7a9092a7053a949893091553c49001e78a3133cdb3` |
| `data/eval/e2synth/inputs.jsonl` | `9c6062d98f27fdbd3d4f5a330fb53879e8a96efcbdccbb0c01a45459b58971be` |
| `data/eval/e2synth/labels.jsonl` | `0afac465c2c2db997d45756004105ebcb85fbd9497fcd8439194cb93c4664feb` |
| `data/eval/e3/inputs.jsonl` | `e06bf6dccd915495dc5a3fed06305dada145d6d6828f7a836ad9e8a9178f6f67` |
| `data/eval/e3/labels.jsonl` | `e3123a14e66aa73136b67f3745454f6e1b4f254bd98cf5b25d3f3591aae8075a` |
| `tokenizer/spm.model` | `1fc208b5b0885b8a1164ea2b6b44303d7acbddd709fab1f6b2dda53231779a37` |
| `tokenizer/freq_src.npy` | `6aaea5b66f82d7c110a49200ca7fa51816cdd4a95ecdbdd89ac0487529bef967` |
| `tokenizer/freq_tgt.npy` | `e5703e051ca13c7564c40f46d6938b09330c7518dc257f68a1c05bb216cb4fed` |
| `official/score.py` | `0e023e486a2a0ca5a2d111b4bf43b0419479b23d4cb79b7789827f35783a8393` |
| `reports/final/main/seg_tuned/diagnostics.json` | `85fc9eac67411a497c68d1dd8effee990ecbf86cbb35334cb7c28e817faec80c` |

Generated files (sha256):

| file | sha256 |
|---|---|
| `eval.json` | `7d5d55a96e857ca316bb9432933fde124e36d05d192a2e7419398402143b99a0` |
| `objective.json` | `8df00c57b5865e27a99bcc73c4f7f143672080d9e8ac6956f7bc5127acbd6389` |
| `diagnostics.json` | `14221f9b590fec2e33f42ba7b62bce658bb009a7f1617c4bb1fabb6ee4aae775` |
| `calibration.json` | `6dc1d4861932d025964f7c64e551400c919f345ff276724396b55d28072d130e` |
