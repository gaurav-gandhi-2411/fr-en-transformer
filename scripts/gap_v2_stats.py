from __future__ import annotations

# EXPLORATORY / POST-HOC (not pre-registered). Statistics for the gap analysis v2: an exact
# Shapley decomposition of the E1-E3 mean-difference over feature GROUPS, a stratified sentence-
# level bootstrap, and collinearity diagnostics. Pure numpy (+ statsmodels for HC3 in the
# coefficient table only); no HF, no model.
#
# Definition of "share of the gap" (same linear form as the pre-registered OLS, nmt/analysis.py):
#   fit  y = a + sum_k b_k x_k + g * D3 + e   (D3 = 1 for E3), OLS, features of the groups in S.
#   With an intercept and D3 in the model the residuals have zero mean inside each domain, so
#       mean(y|E1) - mean(y|E3) = sum_k b_k (xbar_k(E1) - xbar_k(E3)) - g       (exact identity)
#   explained(S) := sum_{k in S} b_k (xbar_k(E1) - xbar_k(E3)) = gap + g(S);  explained({}) = 0
#   (with no features g = -gap).  The Shapley value of group G over the characteristic function
#   explained(.) is its share of the gap in y-units; they sum EXACTLY to explained(all groups),
#   and the remainder  gap - explained(all) = -g(all)  is the unexplained residual (domain).
# A parallel Shapley of R^2 (v(S) = R^2(D3 + S) - R^2(D3 alone)) attributes outcome VARIANCE.
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import statsmodels.api as sm

LABEL = "EXPLORATORY / POST-HOC (not pre-registered)"


def _design(
    groups: Mapping[str, np.ndarray], subset: Sequence[str], domain: np.ndarray
) -> np.ndarray:
    """[const, standardised features of `subset` groups, D3]. Standardising each column makes the
    solve well conditioned and leaves explained(S) unchanged (b_k * delta-mean is scale-free)."""
    cols = [np.ones(len(domain))]
    for g in subset:
        x = np.asarray(groups[g], dtype=float).reshape(len(domain), -1)
        sd = x.std(axis=0)
        sd[sd == 0] = 1.0  # constant column: left at zero after centring, coefficient irrelevant
        cols.append((x - x.mean(axis=0)) / sd)
    cols.append(domain.reshape(-1, 1).astype(float))
    return np.column_stack(cols)


def fit_subset(
    y: np.ndarray, groups: Mapping[str, np.ndarray], subset: Sequence[str], domain: np.ndarray
) -> dict[str, float]:
    """OLS of y on [const, features(subset), D3]; returns explained (see module header), the
    D3 coefficient and R^2. `domain` is 1.0 for E3 and 0.0 for E1."""
    x = _design(groups, subset, domain)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 0.0
    e1, e3 = domain == 0, domain == 1
    explained = 0.0
    col = 1
    for g in subset:
        k = np.asarray(groups[g]).reshape(len(domain), -1).shape[1]
        feats = x[:, col : col + k]
        explained += float(beta[col : col + k] @ (feats[e1].mean(axis=0) - feats[e3].mean(axis=0)))
        col += k
    return {"explained": explained, "gamma": float(beta[-1]), "r2": r2}


def shapley(values: Mapping[frozenset[str], float], players: Sequence[str]) -> dict[str, float]:
    """Exact Shapley values of the characteristic function `values` (keys: every subset of
    `players`, including the empty set). Order-invariant by construction; sums to
    values[all] - values[empty] exactly (up to float rounding)."""
    n = len(players)
    out: dict[str, float] = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for r in range(n):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                total += weight * (values[s | {p}] - values[s])
        out[p] = total
    return out


def all_subset_fits(
    y: np.ndarray, groups: Mapping[str, np.ndarray], domain: np.ndarray
) -> dict[frozenset[str], dict[str, float]]:
    """fit_subset for every subset of the groups (2^G fits; G must be small)."""
    names = list(groups)
    if len(names) > 7:
        raise ValueError("exact Shapley over more than 7 groups is not supported")
    return {
        frozenset(c): fit_subset(y, groups, c, domain)
        for r in range(len(names) + 1)
        for c in itertools.combinations(names, r)
    }


def decompose(
    y: np.ndarray, groups: Mapping[str, np.ndarray], domain: np.ndarray
) -> dict[str, Any]:
    """Gap (E1 mean - E3 mean of y), per-group Shapley share of the gap (y units), unexplained
    residual, and the Shapley split of R^2 over the domain-only baseline."""
    names = list(groups)
    fits = all_subset_fits(y, groups, domain)
    gap = float(y[domain == 0].mean() - y[domain == 1].mean())
    phi = shapley({s: f["explained"] for s, f in fits.items()}, names)
    base_r2 = fits[frozenset()]["r2"]
    phi_r2 = shapley({s: f["r2"] - base_r2 for s, f in fits.items()}, names)
    full = fits[frozenset(names)]
    return {
        "gap": gap,
        "contribution": phi,
        "explained_total": full["explained"],
        "residual": gap - full["explained"],
        "r2_full": full["r2"],
        "r2_domain_only": base_r2,
        "r2_shapley": phi_r2,
    }


def stratified_resample(domain: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sentence-level bootstrap indices, resampling WITHIN each domain so n(E1), n(E3) stay fixed
    (the gap is a between-domain difference; pooled resampling would randomise the group sizes)."""
    out = np.empty(len(domain), dtype=np.int64)
    for d in (0.0, 1.0):
        idx = np.flatnonzero(domain == d)
        out[idx] = rng.choice(idx, size=len(idx), replace=True)
    return out


def bootstrap_decompose(
    y: np.ndarray,
    groups: Mapping[str, np.ndarray],
    domain: np.ndarray,
    n_boot: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Refit the full decomposition on `n_boot` stratified sentence resamples (seed fixed).
    Returns per-quantity arrays (resamples x quantity) and percentile 95% CIs."""
    rng = np.random.default_rng(seed)
    names = list(groups)
    contrib = np.empty((n_boot, len(names)))
    r2c = np.empty((n_boot, len(names)))
    resid = np.empty(n_boot)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = stratified_resample(domain, rng)
        res = decompose(y[idx], {g: np.asarray(groups[g])[idx] for g in names}, domain[idx])
        contrib[b] = [res["contribution"][g] for g in names]
        r2c[b] = [res["r2_shapley"][g] for g in names]
        resid[b], gaps[b] = res["residual"], res["gap"]

    def ci(a: np.ndarray) -> list[float]:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "n_boot": n_boot,
        "seed": seed,
        "contribution_ci": {g: ci(contrib[:, i]) for i, g in enumerate(names)},
        "share_ci": {g: ci(contrib[:, i] / gaps) for i, g in enumerate(names)},
        "residual_ci": ci(resid),
        "residual_share_ci": ci(resid / gaps),
        "gap_ci": ci(gaps),
        "r2_shapley_ci": {g: ci(r2c[:, i]) for i, g in enumerate(names)},
    }


def vif_table(columns: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Variance inflation factor of each column against all others (+ const). inf when a column
    is a perfect linear combination of the others."""
    names = list(columns)
    mat = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    out: dict[str, float] = {}
    for j, n in enumerate(names):
        others = np.delete(mat, j, axis=1)
        a = np.column_stack([np.ones(len(mat)), others])
        beta = np.linalg.lstsq(a, mat[:, j], rcond=None)[0]
        resid = mat[:, j] - a @ beta
        ss_tot = float(((mat[:, j] - mat[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 1.0
        out[n] = float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
    return out


def r2_of(target: np.ndarray, predictors: np.ndarray) -> float:
    """R^2 of an OLS of `target` on `predictors` (+ const)."""
    a = np.column_stack([np.ones(len(target)), np.asarray(predictors, dtype=float)])
    beta = np.linalg.lstsq(a, target, rcond=None)[0]
    resid = target - a @ beta
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 0.0


def collinearity_report(
    groups: Mapping[str, np.ndarray],
    feature_names: Mapping[str, Sequence[str]],
    domain: np.ndarray,
) -> dict[str, Any]:
    """VIFs (features + D3), condition number of the column-standardised design, how well each
    feature / each group predicts the domain dummy (R^2 of D3 on it)."""
    cols: dict[str, np.ndarray] = {}
    for g, arr in groups.items():
        a = np.asarray(arr, dtype=float).reshape(len(domain), -1)
        for j, name in enumerate(feature_names[g]):
            cols[f"{g}.{name}"] = a[:, j]
    cols["domain_e3"] = domain.astype(float)
    mat = np.column_stack(list(cols.values()))
    sd = mat.std(axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mat.mean(axis=0)) / sd
    return {
        "vif": vif_table(cols),
        "condition_number_standardised_design": float(np.linalg.cond(z)),
        "r2_of_domain_on_feature": {
            k: r2_of(domain.astype(float), v) for k, v in cols.items() if k != "domain_e3"
        },
        "r2_of_domain_on_group": {
            g: r2_of(domain.astype(float), np.asarray(a).reshape(len(domain), -1))
            for g, a in groups.items()
        },
    }


def hc3_table(
    y: np.ndarray,
    groups: Mapping[str, np.ndarray],
    feature_names: Mapping[str, Sequence[str]],
    domain: np.ndarray,
) -> dict[str, Any]:
    """Full-model OLS (all groups + D3) on standardised features with HC3 robust SEs
    (statsmodels). Coefficients are per 1 SD of the feature."""
    x = _design(groups, list(groups), domain)
    names = ["const"] + [f"{g}.{n}" for g in groups for n in feature_names[g]] + ["domain_e3"]
    model = sm.OLS(y, x).fit(cov_type="HC3")
    return {
        "n": len(y),
        "r_squared": float(model.rsquared),
        "coef_per_sd": dict(zip(names, (float(v) for v in model.params), strict=True)),
        "se_hc3": dict(zip(names, (float(v) for v in model.bse), strict=True)),
        "p_hc3": dict(zip(names, (float(v) for v in model.pvalues), strict=True)),
    }


def estimand_sensitivity(
    y: np.ndarray, groups: Mapping[str, np.ndarray], domain: np.ndarray
) -> dict[str, float]:
    """Explained part of the E1-E3 gap under alternative estimands (all features together, no
    Shapley): `pooled_with_domain_dummy` (the primary form), `pooled_no_dummy` (common slopes,
    no domain term), `oaxaca_e1_slopes` / `oaxaca_e3_slopes` (per-domain OLS slopes b;
    explained = b . (xbar_E1 - xbar_E3)). Each is also given as a share of the gap."""
    cols = [np.asarray(a, dtype=float).reshape(len(domain), -1) for a in groups.values()]
    x = np.column_stack(cols)
    e1, e3 = domain == 0, domain == 1
    delta = x[e1].mean(axis=0) - x[e3].mean(axis=0)
    gap = float(y[e1].mean() - y[e3].mean())

    def slopes(rows: np.ndarray, with_dummy: bool = False) -> np.ndarray:
        parts = [np.ones(int(rows.sum())), *x[rows].T]
        if with_dummy:
            parts.append(domain[rows])
        beta = np.linalg.lstsq(np.column_stack(parts), y[rows], rcond=None)[0]
        return beta[1 : 1 + x.shape[1]]

    everyone = np.ones(len(domain), dtype=bool)
    explained = {
        "pooled_with_domain_dummy": float(slopes(everyone, True) @ delta),
        "pooled_no_dummy": float(slopes(everyone) @ delta),
        "oaxaca_e1_slopes": float(slopes(e1) @ delta),
        "oaxaca_e3_slopes": float(slopes(e3) @ delta),
    }
    res: dict[str, float] = {"gap": gap}
    for k, v in explained.items():
        res[f"{k}_explained"] = v
        res[f"{k}_share"] = v / gap
    return res
