# Gap analysis v2 -- EXPLORATORY / POST-HOC (not pre-registered)

> **EXPLORATORY / POST-HOC (not pre-registered).** Post-hoc, run after the pre-registered OLS decomposition left almost all of the E1->E3 gap unexplained. The pre-registered result is cited unchanged (`reports/final/main/seg_tuned/analysis.json`, `gap_decomposition`: gap 12.67 chrF, residual 12.88). All features below are heuristics defined in `scripts/gap_v2.py` before results were inspected.

## Findings

1. EXPLORATORY / POST-HOC (not pre-registered). Main's sentence-mean chrF is 12.67 points lower on E3 than on E1 (reproduces the pre-registered gap). The five heuristic feature groups together explain only about 1.1 of those points (about 9%); about 11.6 points (about 91%) stay unexplained. Adding a source-length control group changes this little (explained 0.5 points, sensitivity run).
2. The largest positive group is target-side rarity (about +2.3 points, bootstrap CI in Table 1); alignment noise has the opposite sign (about -1.7) because E1, not E3, has more extreme source/reference length ratios (4.0% vs 2.4% moderate, 1.2% vs 0.6% severe); function/content mix, register markers and subword fertility are each small or indistinguishable from zero (see CIs). The Shapley shares are order-invariant but they inherit the linear, common-slope form of the model.
3. Model-side view (teacher-forced NLL of the reference, a different quantity from chrF): pooled NLL is 1.84 nats per token on E1 and 2.68 on E3. Tokens that are unseen, rare (<10) or in the long tail in the training English side are 2.6% of E3 tokens (0.2% of E1) and carry about 21% of the NLL gap; the other roughly 79% (net of mix and rate terms) sits in the top-1k and 1k-10k bands, i.e. the model assigns lower probability to tokens that are frequent in training, so the gap is mostly not a rare-word effect.
4. E3 is more dialogue- and pronoun-heavy (23% of sources start with a dash or quote vs 5%; reference quote density about 11x), and its French source has higher subword fertility (1.71 vs train 1.54 tokens per word, reproduced here on the source side; English references 1.55 vs train 1.47). These are heuristics; they shift the sets but explain little chrF gap in this model.
5. What the unexplained part is cannot be told from these data: candidates (not tested) include genuine literary-register and archaic-vocabulary difficulty beyond the measured proxies, looser free translations in the E3 references (single reference, chrF), and effects of the linear common-slope form. Only one model (main, step 24,645), one outcome pair (chrF, NLL) and one reference per sentence are used, so nothing here is a causal claim.
6. Collinearity is mild (max VIF 4.5, condition number about 5 on standardised features); no group is near-collinear with the domain dummy (a single group predicts the E3 indicator with R^2 at most 0.11), see `collinearity.json`.

## Table 1 -- Shapley share of the E1-E3 chrF gap (EXPLORATORY / POST-HOC (not pre-registered))

Gap = mean sentence chrF(E1) - mean sentence chrF(E3) = 12.67 (n = 1940 + 1000). Share = Shapley value of the group's mean-difference contribution b_k * (xbar_E1 - xbar_E3) in OLS with a domain dummy (exact over all 2^5 subsets); the groups' contributions plus the residual sum to the gap exactly. 95% CIs: 1000 stratified sentence-bootstrap refits, seed 1234.

| group | chrF points [95% CI] | share of gap [95% CI] | Shapley R^2 [95% CI] |
|---|---|---|---|
| rarity | +2.34 [+1.80, +2.99] | +18.5% [+14.0, +23.8] | 0.0401 [0.0321, 0.0535] |
| function_content | +0.66 [+0.31, +1.05] | +5.2% [+2.5, +8.5] | 0.0103 [0.0050, 0.0193] |
| alignment | -1.74 [-2.31, -1.23] | -13.7% [-19.1, -9.2] | 0.0950 [0.0741, 0.1156] |
| register | -0.17 [-0.86, +0.59] | -1.3% [-7.1, +4.6] | 0.0119 [0.0062, 0.0251] |
| fertility | +0.01 [-0.12, +0.20] | +0.1% [-0.9, +1.6] | 0.0008 [0.0004, 0.0052] |
| **explained (all groups)** | +1.10 | +8.7% | R^2 0.2297 (domain-only 0.0716) |
| **unexplained residual (domain)** | +11.57 [+10.00, +13.03] | +91.3% [+82.7, +100.8] | - |

Sensitivity (adds a `length` control group, log source words; not part of (a)-(e)): explained +0.50 chrF (+3.9%), residual +12.17 [+10.60, +13.65]; length -0.58 [-0.86, -0.35].

## Table 2 -- Token NLL by training-frequency band (EXPLORATORY / POST-HOC (not pre-registered))

Teacher-forced NLL of the reference (fp32, eval mode, no label smoothing, EOS excluded). Pooled token gap E3-E1 = 0.846 nats (mix + rate terms sum to it exactly).

| band | token share E1 | token share E3 | mean NLL E1 | mean NLL E3 | mix term | rate term |
|---|---|---|---|---|---|---|
| unseen | 0.0% | 0.3% | 9.38 | 9.73 | +0.023 | +0.001 |
| rare | 0.1% | 1.3% | 5.06 | 7.81 | +0.062 | +0.036 |
| top1k | 68.0% | 68.6% | 1.56 | 2.16 | +0.009 | +0.411 |
| mid_1k_10k | 31.8% | 28.8% | 2.42 | 3.52 | -0.072 | +0.316 |
| tail_ge10 | 0.1% | 1.0% | 3.38 | 6.13 | +0.031 | +0.029 |

## Figure (EXPLORATORY / POST-HOC (not pre-registered))

![EXPLORATORY / POST-HOC (not pre-registered): gap components](gap_v2_components.png)

*EXPLORATORY / POST-HOC (not pre-registered).* Left: Shapley contribution of each feature group to the chrF gap with 95% bootstrap CIs; the grey bar is what remains unexplained. Right: descriptive split of the pooled token-NLL gap by training-frequency band.

## Files

`features_summary.json` (feature means/sd per domain, novelty, fertility, alignment, word lists), `nll_breakdown.json`, `gap_shares.json`, `ols_hc3.json`, `collinearity.json`, `provenance.json`. All carry the key `label`.

## Provenance

- code commit (the commit the numbers were generated from): `1294f52fb03e7bebb1a3753aea5179f5ff975cf6` (tree dirty: False)
- eval repo `OWNER/fr-en-transformer-eval` @ `c3d8598252853fcd7df1ef4a00e8b0382b8f4351` (private=True, manifest sha256 `613ea8c0d2e22958e6ef5fc55fe63c8d8f6c218c63958ed5edeb5b33b5ca37d1`, model files verified: True)
- data repo `OWNER/fr-en-transformer-data` @ `c40e393740f41dd3aac3e615952ac7b86d4e58e0` (private=True; shard + spm sha256 verified against `data/shards/manifest.json`)
- word list sha256 `ce42d239b492e7830b20dde1fd8c0d712dffeb9526abc9b81a66bc702c58fa5d`; runtime {'verify': 0.01, 'data_chrf': 6.57, 'train_counts': 17.48, 'features': 1.05, 'nll': 51.51, 'nll_breakdown': 1.14, 'decomposition': 226.38, 'total': 303.31}
- input file sha256 values: `provenance.json` (`input_files_sha256`, `eval_model_sha256`, `data_repo_files_sha256`)
- line endings: LF (files are written as bytes)
