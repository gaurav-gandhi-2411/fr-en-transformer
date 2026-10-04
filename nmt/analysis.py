from __future__ import annotations

# Automatic error taxonomy + generalization-gap decomposition: per-sentence features, OLS gap
# decomposition with HC3 robust SEs, metric-artifact share, figures, and 6 auto-selected worst-
# chrF examples. No manual labeling, no LLM judges.
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this module never opens a display window
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import statsmodels.api as sm

from nmt.data.e2synth import BUCKET_NAMES as E2SYNTH_BUCKET_NAMES
from nmt.data.normalize import normalize_text
from nmt.evaluate import load_official_module, load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "tokenizer" / "spm.model"
DEFAULT_FREQ_SRC_PATH = REPO_ROOT / "tokenizer" / "freq_src.npy"
DEFAULT_FREQ_TGT_PATH = REPO_ROOT / "tokenizer" / "freq_tgt.npy"

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
_PUNCT_RE = re.compile(r"[.,;:!?\"'()\-—–«»]")
_CAP_WORD_RE = re.compile(r"^[A-ZÀ-Ý][\w'-]*$", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# -------------------------------------------------------------------------------------------
# Per-sentence features
# -------------------------------------------------------------------------------------------


def length_ratio(hyp: str, ref: str) -> float:
    """hyp/ref word-count ratio. 1.0 for an empty/empty pair (no evidence of a length problem);
    a large finite sentinel (not inf, to stay OLS-friendly) when only the reference is empty."""
    h, r = len(_words(hyp)), len(_words(ref))
    if r == 0:
        return 1.0 if h == 0 else 10.0
    return h / r


def repetition_rate(hyp: str, n: int = 3) -> float:
    """Share of duplicate n-grams in the hypothesis (0.0 if fewer than n words exist)."""
    grams = _ngrams(_words(hyp), n)
    if not grams:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    return duplicates / len(grams)


def is_truncated(hyp: str, ref: str) -> bool:
    """True when hyp/ref length ratio is below 0.5."""
    h, r = len(_words(hyp)), len(_words(ref))
    return r > 0 and h < 0.5 * r


def untranslated_copy_rate(
    hyp: str, source: str, sp: spm.SentencePieceProcessor, freq_tgt: np.ndarray
) -> float:
    """Share of hyp subword tokens that also occur in the source AND were never observed on the
    train target side (freq_tgt == 0) -- i.e. copied straight through rather than translated."""
    hyp_ids = sp.encode(hyp, out_type=int)
    if not hyp_ids:
        return 0.0
    src_ids = set(sp.encode(source, out_type=int))
    copied = sum(1 for t in hyp_ids if t in src_ids and freq_tgt[t] == 0)
    return copied / len(hyp_ids)


def source_rarity(
    source: str, sp: spm.SentencePieceProcessor, freq_src: np.ndarray, byte_fallback_ids: set[int]
) -> dict[str, float]:
    """Mean/min train frequency of the source's subwords, and its byte-fallback token rate."""
    ids = sp.encode(source, out_type=int)
    if not ids:
        return {"mean_freq": 0.0, "min_freq": 0.0, "byte_fallback_rate": 0.0}
    freqs = [int(freq_src[i]) for i in ids]
    n_byte = sum(1 for i in ids if i in byte_fallback_ids)
    return {
        "mean_freq": float(np.mean(freqs)),
        "min_freq": float(np.min(freqs)),
        "byte_fallback_rate": n_byte / len(ids),
    }


def proper_nouns(source: str) -> list[str]:
    """Heuristic proper-noun detector: non-initial capitalized source words."""
    words = source.split()
    return [w for i, w in enumerate(words) if i > 0 and _CAP_WORD_RE.match(w)]


def proper_noun_copy_accuracy(source: str, hyp: str) -> float | None:
    """Share of `proper_nouns(source)` found verbatim (case-insensitive) in `hyp`. `None` when
    the source has no proper nouns to check (excluded from the feature matrix as missing, not
    imputed as 0 or 1)."""
    nouns = proper_nouns(source)
    if not nouns:
        return None
    hyp_lower = hyp.lower()
    return sum(1 for n in nouns if n.lower() in hyp_lower) / len(nouns)


def dialogue_punct_density(source: str) -> float:
    """Punctuation characters per source character (dashes/quotes are the dominant dialogue
    markers in literary French, so this doubles as a dialogue-density proxy)."""
    if not source:
        return 0.0
    return len(_PUNCT_RE.findall(source)) / len(source)


@dataclass
class SentenceFeatures:
    id: str
    domain: str  # "e1" | "e3" | "dev"
    length_ratio: float
    repetition_rate: float
    truncated: bool
    untranslated_copy_rate: float
    src_rarity_mean: float
    src_rarity_min: float
    src_byte_fallback_rate: float
    proper_noun_copy_accuracy: float | None
    dialogue_punct_density: float
    chrf: float


def build_sentence_features(
    rows: list[dict[str, str]],
    sp: spm.SentencePieceProcessor,
    freq_src: np.ndarray,
    freq_tgt: np.ndarray,
    byte_fallback_ids: set[int],
    domain: str,
) -> list[SentenceFeatures]:
    """`rows`: `[{"id", "source", "reference", "hyp"}, ...]`."""
    module = load_official_module()
    out = []
    for r in rows:
        hyp, ref, source = r["hyp"], r["reference"], r["source"]
        rarity = source_rarity(source, sp, freq_src, byte_fallback_ids)
        out.append(
            SentenceFeatures(
                id=r["id"],
                domain=domain,
                length_ratio=length_ratio(hyp, ref),
                repetition_rate=repetition_rate(hyp),
                truncated=is_truncated(hyp, ref),
                untranslated_copy_rate=untranslated_copy_rate(hyp, source, sp, freq_tgt),
                src_rarity_mean=rarity["mean_freq"],
                src_rarity_min=rarity["min_freq"],
                src_byte_fallback_rate=rarity["byte_fallback_rate"],
                proper_noun_copy_accuracy=proper_noun_copy_accuracy(source, hyp),
                dialogue_punct_density=dialogue_punct_density(source),
                chrf=float(module.chrf_sentence(hyp, ref)),
            )
        )
    return out


# -------------------------------------------------------------------------------------------
# Gap decomposition
# -------------------------------------------------------------------------------------------


def gap_decomposition(features: list[SentenceFeatures]) -> dict[str, Any]:
    """OLS of sentence chrF on [|length_ratio - 1|, repetition_rate, -log1p(src_rarity_mean),
    dialogue_punct_density, domain_indicator] over E1 (domain="e1") union E3 (domain="e3"), HC3
    robust SEs.

    Gap share attributable to each covariate = its OLS coefficient * (E1 mean - E3 mean) of that
    covariate -- a single-regression linear decomposition (not a full two-group Oaxaca-Blinder
    with separate slopes per domain; documented as a simplification appropriate at this sample
    size). The "residual domain" share is `-coef(domain_indicator)`: the part of the E1-E3 gap
    the other covariates do not explain.
    """
    e1e3 = [f for f in features if f.domain in ("e1", "e3")]
    y = np.array([f.chrf for f in e1e3])
    domain = np.array([1.0 if f.domain == "e3" else 0.0 for f in e1e3])
    length_dev = np.array([abs(f.length_ratio - 1.0) for f in e1e3])
    repetition = np.array([f.repetition_rate for f in e1e3])
    rarity = np.array([-np.log1p(f.src_rarity_mean) for f in e1e3])
    dialogue = np.array([f.dialogue_punct_density for f in e1e3])

    covariate_names = ["length_dev", "repetition_rate", "rarity", "dialogue_punct_density"]
    covariates = [length_dev, repetition, rarity, dialogue]
    x = np.column_stack([*covariates, domain])
    # has_constant="add": statsmodels' default ("skip") silently omits the intercept column
    # whenever any OTHER covariate happens to be constant across the sample (e.g. a slice with
    # zero measured repetition) -- that would shift every downstream index by one and desync the
    # `names` list below. Force the intercept to always be added.
    model = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HC3")
    names = ["const", *covariate_names, "domain_e3"]
    coefs = dict(zip(names, (float(v) for v in model.params), strict=True))
    ses = dict(zip(names, (float(v) for v in model.bse), strict=True))
    pvals = dict(zip(names, (float(v) for v in model.pvalues), strict=True))

    mean_e1 = {
        n: float(c[domain == 0].mean()) for n, c in zip(covariate_names, covariates, strict=True)
    }
    mean_e3 = {
        n: float(c[domain == 1].mean()) for n, c in zip(covariate_names, covariates, strict=True)
    }
    total_gap = float(y[domain == 0].mean() - y[domain == 1].mean()) if len(e1e3) else 0.0

    contributions = {n: coefs[n] * (mean_e1[n] - mean_e3[n]) for n in covariate_names}
    contributions["residual_domain"] = -coefs["domain_e3"]
    share_of_gap = {n: (v / total_gap if total_gap != 0 else 0.0) for n, v in contributions.items()}

    return {
        "n": len(e1e3),
        "total_gap_chrf_e1_minus_e3": total_gap,
        "coefficients": coefs,
        "robust_se_HC3": ses,
        "p_values": pvals,
        "contributions_chrf": contributions,
        "share_of_gap": share_of_gap,
        "r_squared": float(model.rsquared),
    }


# -------------------------------------------------------------------------------------------
# Metric-artifact share
# -------------------------------------------------------------------------------------------


def normalize_reference_for_artifact_check(ref: str) -> str:
    """Unify apostrophes/quotes (`nmt.data.normalize.normalize_text`) and additionally strip a
    stray trailing quote character. References only -- outputs are never tuned to match
    this."""
    r = normalize_text(ref).rstrip()
    while r and r[-1] in "\"'":
        r = r[:-1].rstrip()
    return r


def metric_artifact_share(rows: list[dict[str, str]]) -> dict[str, Any]:
    """`rows`: `[{"id", "hyp", "reference"}, ...]`. Rescores after normalizing references only;
    the delta is the part of the seen/unseen gap attributable to reference noise rather than the
    model (a diagnostic -- never used to tune outputs)."""
    module = load_official_module()
    hyps = [r["hyp"] for r in rows]
    refs = [r["reference"] for r in rows]
    refs_norm = [normalize_reference_for_artifact_check(r) for r in refs]
    original = module.score_slice(hyps, refs)
    renormalized = module.score_slice(hyps, refs_norm)
    if original is None:
        return {"n": 0, "chrf_original": None, "chrf_ref_normalized": None, "delta_chrf": 0.0}
    return {
        "n": len(rows),
        "chrf_original": original["chrf"],
        "chrf_ref_normalized": renormalized["chrf"],
        "delta_chrf": renormalized["chrf"] - original["chrf"],
    }


# -------------------------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------------------------

_LENGTH_BUCKET_ORDER = ("<=10", "11-20", "21-40", "41-80", ">80")


def _e2synth_chrf_by_bucket(ev: dict[str, Any]) -> dict[str, dict[str, float]]:
    """E2-synth's per-char-bucket chrF (point + bootstrap CI) from one `eval.json`, in bucket
    order; empty when the run has no E2-synth entry. Read from `official_ci_by_slice` (E2-synth's
    slice names are its char buckets), never from `length_buckets_e1_e2_e3`."""
    ci = ev.get("sets", {}).get("e2synth", {}).get("official_ci_by_slice", {}).get("chrf", {})
    return {b: ci[b] for b in E2SYNTH_BUCKET_NAMES if b in ci}


def figure_chrf_vs_length_bucket(eval_jsons: dict[str, dict[str, Any]], out_path: Path) -> Path:
    """One line per run (accepts multiple `eval.json`s so S1 vs S2 can be overlaid),
    reading each run's `length_buckets_e1_e2_e3` view. If any run carries an E2-synth entry, a
    second, separately labelled panel plots E2-synth (synthetic; reuses E2 sentences) chrF by
    French char bucket with bootstrap CIs -- never mixed into the E1+E2+E3 word-bucket panel."""
    synth = {label: _e2synth_chrf_by_bucket(ev) for label, ev in eval_jsons.items()}
    synth = {label: buckets for label, buckets in synth.items() if buckets}
    if synth:
        fig, (ax, ax_synth) = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig, ax = plt.subplots()
    for label, ev in eval_jsons.items():
        buckets = ev.get("length_buckets_e1_e2_e3", {})
        xs = [b for b in _LENGTH_BUCKET_ORDER if b in buckets]
        ys = [buckets[b]["chrf"] for b in xs]
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("source length bucket (words)")
    ax.set_ylabel("chrF")
    ax.set_title("chrF vs source length (E1+E2+E3)")
    if eval_jsons:
        ax.legend()
    if synth:
        for label, buckets in synth.items():
            xs = list(buckets)
            ys = [buckets[b]["point"] for b in xs]
            yerr = [
                [buckets[b]["point"] - buckets[b]["ci_low"] for b in xs],
                [buckets[b]["ci_high"] - buckets[b]["point"] for b in xs],
            ]
            ax_synth.errorbar(
                [b.removeprefix("e2synth_").replace("_", "-") for b in xs],
                ys,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=label,
            )
        ax_synth.set_xlabel("source length bucket (French characters)")
        ax_synth.set_ylabel("chrF")
        ax_synth.set_title("E2-synth (synthetic): chrF vs source length\n(reuses E2 sentences)")
        ax_synth.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_chrf_vs_rarity_decile(features: list[SentenceFeatures], out_path: Path) -> Path:
    """Mean chrF per source-rarity decile (decile 1 = most common source subwords, 10 = rarest)."""
    fig, ax = plt.subplots()
    if features:
        rarities = np.array([f.src_rarity_mean for f in features])
        chrfs = np.array([f.chrf for f in features])
        order = np.argsort(rarities)
        bins = np.array_split(order, 10)
        xs = [i + 1 for i, b in enumerate(bins) if len(b)]
        ys = [float(chrfs[b].mean()) for b in bins if len(b)]
        ax.plot(xs, ys, marker="o")
    ax.set_xlabel("source rarity decile (1=common, 10=rare)")
    ax.set_ylabel("mean chrF")
    ax.set_title("chrF vs source rarity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


_FAILURE_MODES = ("truncated", "repetition", "untranslated_copy")


def figure_failure_mode_rates(features: list[SentenceFeatures], out_path: Path) -> Path:
    """Bar chart of failure-mode rates (truncation, any 3-gram repetition, meaningful
    untranslated-copy rate) per domain/slice."""
    domains = sorted({f.domain for f in features})
    fig, ax = plt.subplots()
    x = np.arange(len(_FAILURE_MODES))
    width = 0.8 / max(1, len(domains))
    for i, d in enumerate(domains):
        subset = [f for f in features if f.domain == d]
        n = len(subset) or 1
        rates = [
            sum(1 for f in subset if f.truncated) / n,
            sum(1 for f in subset if f.repetition_rate > 0) / n,
            sum(1 for f in subset if f.untranslated_copy_rate > 0.1) / n,
        ]
        ax.bar(x + i * width, rates, width=width, label=d)
    ax.set_xticks(x + width * (len(domains) - 1) / 2)
    ax.set_xticklabels(_FAILURE_MODES)
    ax.set_ylabel("rate")
    ax.set_title("Failure-mode rates per slice")
    if domains:
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


_UNTRANSLATED_COPY_THRESHOLD = 0.1  # same cut as `_failure_type` / the failure-mode figure
_OVERLONG_RATIO = 1.5  # hyp/ref word ratio above which a hypothesis counts as over-generated


def failure_mode_rates(features: list[SentenceFeatures]) -> dict[str, Any]:
    """Failure-mode rates of one group of sentences (a slice, a split): the share with any
    repeated 3-gram, truncated (hyp/ref < 0.5), untranslated copy (> 10% of hyp subwords copied
    from the source and never seen on the train target side), and over-long (hyp/ref > 1.5), plus
    the mean and median per-sentence hyp/ref word-count ratio. Rates are shares in [0, 1]; an empty
    group yields n=0 and None rates (not 0, which would read as 'no failures')."""
    n = len(features)
    if n == 0:
        keys = ("repetition", "truncation", "untranslated_copy", "overlong")
        return {
            "n": 0,
            **{f"{k}_rate": None for k in keys},
            "length_ratio_mean": None,
            "length_ratio_median": None,
        }
    ratios = np.array([f.length_ratio for f in features])
    return {
        "n": n,
        "repetition_rate": sum(1 for f in features if f.repetition_rate > 0) / n,
        "truncation_rate": sum(1 for f in features if f.truncated) / n,
        "untranslated_copy_rate": sum(
            1 for f in features if f.untranslated_copy_rate > _UNTRANSLATED_COPY_THRESHOLD
        )
        / n,
        "overlong_rate": float(np.mean(ratios > _OVERLONG_RATIO)),
        "length_ratio_mean": float(ratios.mean()),
        "length_ratio_median": float(np.median(ratios)),
    }


def rarity_bucket_edges(rarities: list[float], n_buckets: int = 5) -> list[float]:
    """Interior quantile edges (n_buckets - 1 values) of the source-rarity feature. Computed on the
    sources only, so every model is bucketed identically."""
    qs = [i / n_buckets for i in range(1, n_buckets)]
    return [float(v) for v in np.quantile(np.array(rarities), qs)]


def rarity_bucket_chrf(
    features: list[SentenceFeatures], edges: list[float]
) -> list[dict[str, Any]]:
    """Mean sentence chrF per source-rarity bucket. `src_rarity_mean` is a mean train frequency, so
    a LOW value is a rare source; bucket 1 is the most common sources (highest frequency) and the
    last bucket the rarest. A value equal to an edge falls in the lower-frequency bucket
    (`np.searchsorted(..., side="left")`)."""
    n_buckets = len(edges) + 1
    by_bucket: list[list[SentenceFeatures]] = [[] for _ in range(n_buckets)]
    for f in features:
        ascending = int(np.searchsorted(edges, f.src_rarity_mean, side="left"))
        by_bucket[n_buckets - 1 - ascending].append(f)
    return [
        {
            "bucket": i + 1,
            "n": len(sel),
            "mean_src_freq": float(np.mean([f.src_rarity_mean for f in sel])) if sel else None,
            "chrf": float(np.mean([f.chrf for f in sel])) if sel else None,
        }
        for i, sel in enumerate(by_bucket)
    ]


# -------------------------------------------------------------------------------------------
# Example selection
# -------------------------------------------------------------------------------------------


def _failure_type(f: SentenceFeatures) -> str:
    if f.truncated:
        return "truncated"
    if f.repetition_rate > 0:
        return "repetition"
    if f.untranslated_copy_rate > 0.1:
        return "untranslated_copy"
    return "low_chrf_other"


def select_worst_examples(
    dev_features: list[SentenceFeatures],
    dev_rows_by_id: dict[str, dict[str, str]],
    n_per_slice: int = 2,
) -> list[dict[str, Any]]:
    """6 examples (2 per dev slice), the worst-chrF sentences in each slice, auto-tagged with a
    failure type."""
    by_slice: dict[str, list[SentenceFeatures]] = defaultdict(list)
    for f in dev_features:
        by_slice[dev_rows_by_id[f.id]["slice"]].append(f)
    examples = []
    for slice_name in sorted(by_slice):
        worst = sorted(by_slice[slice_name], key=lambda f: f.chrf)[:n_per_slice]
        for f in worst:
            row = dev_rows_by_id[f.id]
            examples.append(
                {
                    "id": f.id,
                    "slice": slice_name,
                    "failure_type": _failure_type(f),
                    "chrf": f.chrf,
                    "source": row["source"],
                    "reference": row["reference"],
                    "hypothesis": row["hyp"],
                }
            )
    return examples


# -------------------------------------------------------------------------------------------
# Top-level orchestration
# -------------------------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    freq_src_path: Path = DEFAULT_FREQ_SRC_PATH
    freq_tgt_path: Path = DEFAULT_FREQ_TGT_PATH


def _rows_with_predictions(split: str, predictions: dict[str, str]) -> list[dict[str, str]]:
    inputs, labels = load_split(split)
    label_by_id = {r["id"]: r for r in labels}
    rows = []
    for r in inputs:
        rid = r["id"]
        if rid not in predictions:
            continue
        rows.append(
            {
                "id": rid,
                "source": r["source"],
                "reference": label_by_id[rid]["reference"],
                "hyp": predictions[rid],
                "slice": label_by_id[rid].get("slice", split),
            }
        )
    return rows


def run_analysis(
    e1_predictions: dict[str, str],
    e3_predictions: dict[str, str],
    dev_predictions: dict[str, str],
    out_dir: Path,
    other_eval_jsons: dict[str, dict[str, Any]] | None = None,
    cfg: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build features over E1 union E3, fit the gap-decomposition OLS, compute the metric-
    artifact share (dev's unseen_domain slice, and E3), render the three figures, select
    6 worst-chrF dev examples, and write `analysis.json` + `examples.json` + `figures/*.png`
    under `out_dir`.
    """
    cfg = cfg or AnalysisConfig()
    sp = spm.SentencePieceProcessor()
    sp.load(str(cfg.tokenizer_path))
    freq_src = np.load(cfg.freq_src_path)
    freq_tgt = np.load(cfg.freq_tgt_path)
    byte_fallback_ids = {
        i for i in range(sp.get_piece_size()) if _BYTE_FALLBACK_RE.match(sp.id_to_piece(i))
    }

    e1_rows = _rows_with_predictions("e1", e1_predictions)
    e3_rows = _rows_with_predictions("e3", e3_predictions)
    dev_rows = _rows_with_predictions("dev", dev_predictions)

    e1_features = build_sentence_features(e1_rows, sp, freq_src, freq_tgt, byte_fallback_ids, "e1")
    e3_features = build_sentence_features(e3_rows, sp, freq_src, freq_tgt, byte_fallback_ids, "e3")
    dev_features = build_sentence_features(
        dev_rows, sp, freq_src, freq_tgt, byte_fallback_ids, "dev"
    )
    all_features = e1_features + e3_features

    decomposition = gap_decomposition(all_features)

    dev_unseen_rows = [r for r in dev_rows if r["slice"] == "unseen_domain"]
    artifact = {
        "dev_unseen_domain": metric_artifact_share(dev_unseen_rows),
        "e3": metric_artifact_share(e3_rows),
    }

    figures_dir = out_dir / "figures"
    fig_paths = {
        "chrf_vs_length_bucket": str(
            figure_chrf_vs_length_bucket(
                other_eval_jsons or {}, figures_dir / "chrf_vs_length_bucket.png"
            )
        ),
        "chrf_vs_rarity_decile": str(
            figure_chrf_vs_rarity_decile(all_features, figures_dir / "chrf_vs_rarity_decile.png")
        ),
        "failure_mode_rates": str(
            figure_failure_mode_rates(all_features, figures_dir / "failure_mode_rates.png")
        ),
    }

    dev_rows_by_id = {r["id"]: r for r in dev_rows}
    examples = select_worst_examples(dev_features, dev_rows_by_id)

    analysis: dict[str, Any] = {
        "n_e1": len(e1_rows),
        "n_e3": len(e3_rows),
        "n_dev": len(dev_rows),
        "gap_decomposition": decomposition,
        "metric_artifact_share": artifact,
        "figures": fig_paths,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return analysis
