from __future__ import annotations

# ruff: noqa: E501
# (E501: the report text and f-string table rows are kept as readable single lines)
# Builds the derived artifacts and SUMMARY.md of reports/final from what `scripts.eval_local`
# (and `scripts.comet_final`) already wrote there. Reads only files under --out-root; never
# decodes, never selects, never edits official/score.py.
#
# Writes, under <out-root>:
#   <run>/<variant>/diagnostics.json   failure-mode rates per slice, chrF by source-rarity bucket
#   <run>/<variant>/comet_summary.json COMET-22 mean + 95% bootstrap CI per set / slice
#   compare/extras.json                paired tests PREREG lists as "also reported" (>80-word bucket)
#   sanity.json                        scorer parity / re-score / selection-objective checks
#   SUMMARY.md, EXAMPLES.md            the paste-ready report (every figure printed from a JSON above)
#
# CLI: `python -m scripts.build_final_report --out-root reports/final`
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

from nmt.analysis import (
    _BYTE_FALLBACK_RE,
    DEFAULT_FREQ_SRC_PATH,
    DEFAULT_FREQ_TGT_PATH,
    DEFAULT_TOKENIZER_PATH,
    _rows_with_predictions,
    _words,
    build_sentence_features,
    failure_mode_rates,
    length_ratio,
    rarity_bucket_chrf,
    rarity_bucket_edges,
    repetition_rate,
)
from nmt.compare import _objective_inputs, _objective_point
from nmt.evaluate import (
    _official_metric_fn,
    length_bucket_label,
    load_split,
    paired_bootstrap,
    run_official_scorer_cli,
)
from scripts.eval_local import HYPOTHESES, RUNS, SPLITS, VARIANTS, gold_path

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP, SEED = 1000, 1234
BASELINE_RUN = (
    "baseline_copy_source"  # copy-the-source floor, written by scripts.copy_source_baseline
)
PINNED_REVISIONS = {
    "main": "c3d8598252853fcd7df1ef4a00e8b0382b8f4351",
    "s1_sin_l4": "041d49269f61d0cbd9c0380a4f0a0a31f7599547",
    "s2_rope_l4": "bdd850a6d384e088dd1eb52d67ad767149dee3af",
    "s3_rope_concat_l4": "ac92b8da971fdf64974089f40e8f4ddbf9d9638b",
}
RUN_LABEL = {
    "main": "main (deep-enc/shallow-dec, 24,645 steps)",
    "s1_sin_l4": "S1 sinusoidal (4,107 steps)",
    "s2_rope_l4": "S2 RoPE (4,107 steps)",
    "s3_rope_concat_l4": "S3 RoPE + concat augmentation (4,107 steps)",
}
SET_LABEL = {
    "dev": "official dev",
    "e1": "E1 seen-proxy",
    "e2": "E2 long-proxy",
    "e2synth": "E2-synth (synthetic)",
    "e3": "E3 books-proxy",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, obj: Any) -> None:
    # newline="\n": the repo's .gitattributes stores text as LF, so hashes recorded here must be
    # hashes of LF bytes or they would not match a fresh checkout
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -------------------------------------------------------------------------------------------
# formatting helpers
# -------------------------------------------------------------------------------------------


def fmt_p(p: float, n_resamples: int = N_BOOTSTRAP) -> str:
    """One-sided bootstrap p; 0 resamples in the tail is reported as '<1/n', never as 0."""
    return f"< {1 / n_resamples:g}" if p == 0 else f"{p:.3f}"


def fmt_ci(point: float, lo: float, hi: float, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{point:{sign}.2f} [{lo:{sign}.2f}, {hi:{sign}.2f}]"


def bootstrap_mean_ci(
    values: list[float], n_resamples: int = N_BOOTSTRAP, seed: int = SEED
) -> dict[str, float]:
    """Mean with a 95% percentile bootstrap CI (same index convention as nmt.evaluate: one
    default_rng(seed) matrix of (n_resamples, n) draws)."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = np.sort(arr[idx].mean(axis=1))
    return {
        "point": float(arr.mean()),
        "ci_low": float(means[int(0.025 * n_resamples)]),
        "ci_high": float(means[min(n_resamples - 1, int(0.975 * n_resamples))]),
        "n": len(arr),
        "n_resamples": n_resamples,
    }


def slice_means(scores: list[float], slices: list[str]) -> dict[str, dict[str, float]]:
    """Bootstrap mean per slice label (scores and slices are aligned per sentence)."""
    out: dict[str, dict[str, float]] = {}
    for name in sorted(set(slices)):
        out[name] = bootstrap_mean_ci(
            [s for s, sl in zip(scores, slices, strict=True) if sl == name]
        )
    return out


# -------------------------------------------------------------------------------------------
# derived JSONs
# -------------------------------------------------------------------------------------------


def _resources() -> tuple[spm.SentencePieceProcessor, np.ndarray, np.ndarray, set[int]]:
    sp = spm.SentencePieceProcessor()
    sp.load(str(DEFAULT_TOKENIZER_PATH))
    byte_ids = {i for i in range(sp.get_piece_size()) if _BYTE_FALLBACK_RE.match(sp.id_to_piece(i))}
    return sp, np.load(DEFAULT_FREQ_SRC_PATH), np.load(DEFAULT_FREQ_TGT_PATH), byte_ids


def normalize_and_reindex(out_root: Path) -> int:
    """eval_local writes text with the platform newline (CRLF on Windows) while git stores LF
    (.gitattributes `text=auto eol=lf`), so the sha256s it recorded would not match a checkout.
    Rewrites every non-downloaded text file under `out_root` with LF and refreshes the sha256/bytes
    of every entry of the run and compare `index.json` files. Returns the number of files
    converted. `source/` (verbatim HF bytes) is never touched."""
    converted = 0
    for f in sorted(out_root.rglob("*")):
        if (
            not f.is_file()
            or f.suffix not in (".json", ".md")
            or "source" in f.relative_to(out_root).parts
        ):
            continue
        raw = f.read_bytes()
        if b"\r\n" in raw:
            f.write_bytes(raw.replace(b"\r\n", b"\n"))
            converted += 1
    for index_path in [
        *(out_root / r / "index.json" for r in RUNS),
        out_root / "compare" / "index.json",
    ]:
        if not index_path.is_file():
            continue
        index = _read(index_path)
        for art in index["artifacts"]:
            target = index_path.parent / art["path"]
            art["sha256"], art["bytes"] = sha256_file(target), target.stat().st_size
        _write(index_path, index)
    return converted


COPY_MIN_LEN = 4  # shorter words ("de", "la", "en") are too often legitimately shared


def word_copies(hyp: str, source: str, ref: str) -> list[str]:
    """Heuristic untranslated-copy words: hypothesis words of length >= COPY_MIN_LEN that equal a
    source word (case-insensitive) and do not occur in the reference. Names and numbers the
    reference also keeps are therefore excluded; a name the reference spells differently is not."""
    src_words = {w.lower() for w in _words(source)}
    ref_words = {w.lower() for w in _words(ref)}
    return [
        w
        for w in _words(hyp)
        if len(w) >= COPY_MIN_LEN and w.lower() in src_words and w.lower() not in ref_words
    ]


def pick_rule_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two deterministic, rule-based examples from `rows` (dicts with id, source, reference, hyp):
    (A) the largest |hyp/ref word ratio - 1| among rows whose reference has >= 5 words;
    (B) the highest hypothesis repeated-3-gram share among rows whose REFERENCE has none and that
    is not A. Ties go to the smaller id. No chrF, no manual choice."""
    ranked_a = sorted(
        (r for r in rows if len(_words(r["reference"])) >= 5),
        key=lambda r: (-abs(length_ratio(r["hyp"], r["reference"]) - 1.0), r["id"]),
    )
    out = []
    if ranked_a:
        a = ranked_a[0]
        out.append({**a, "rule": "A: largest length-ratio deviation (ref >= 5 words)"})
        ranked_b = sorted(
            (
                r
                for r in rows
                if r["id"] != a["id"]
                and repetition_rate(r["reference"]) == 0
                and repetition_rate(r["hyp"]) > 0
            ),
            key=lambda r: (-repetition_rate(r["hyp"]), r["id"]),
        )
        if ranked_b:
            out.append(
                {**ranked_b[0], "rule": "B: highest hypothesis repetition (reference has none)"}
            )
    for e in out:
        e["length_ratio"] = length_ratio(e["hyp"], e["reference"])
        e["hyp_repetition_rate"] = repetition_rate(e["hyp"])
    return out


def _reference_side_and_word_copy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reference-side repeated-3-gram share and the word-level copy heuristic for one group."""
    n = len(rows)
    copies = [word_copies(r["hyp"], r["source"], r["reference"]) for r in rows]
    return {
        "ref_repetition_rate": sum(1 for r in rows if repetition_rate(r["reference"]) > 0) / n,
        "word_copy_sentence_rate": sum(1 for c in copies if c) / n,
        "word_copy_sentences": sum(1 for c in copies if c),
        "word_copy_words": sum(len(c) for c in copies),
    }


def build_diagnostics(
    out_root: Path, run: str, variant: str, res: Any, base_dir: Path | None = None
) -> dict[str, Any]:
    """Failure-mode rates per set and per slice, and chrF by source-rarity quintile (pooled
    E1+E2+E3, edges from the sources only), for one (run, variant). `base_dir` overrides the
    default `<out_root>/<run>/<variant>` (the copy-source baseline has no variant level)."""
    sp, freq_src, freq_tgt, byte_ids = res
    base = base_dir if base_dir is not None else out_root / run / variant
    groups: dict[str, list] = {}
    group_rows: dict[str, list] = {}
    pooled = []
    pooled_rows: list = []
    for split in SPLITS:
        preds = _read(base / f"{split}_predictions.json")
        rows = _rows_with_predictions(split, preds)
        feats = build_sentence_features(rows, sp, freq_src, freq_tgt, byte_ids, split)
        groups[split] = feats
        group_rows[split] = rows
        slice_of = {r["id"]: r["slice"] for r in rows}
        names = sorted(set(slice_of.values()))
        if len(names) > 1:
            for name in names:
                groups[f"{split}:{name}"] = [f for f in feats if slice_of[f.id] == name]
                group_rows[f"{split}:{name}"] = [r for r in rows if r["slice"] == name]
        if split in ("dev", "e1", "e2", "e3"):
            pooled_rows += rows
        if split in ("e1", "e2", "e3"):
            pooled += feats
    edges = rarity_bucket_edges([f.src_rarity_mean for f in pooled])
    out = {
        "run": run,
        "variant": variant,
        "failure_modes": {
            g: {**failure_mode_rates(f), **_reference_side_and_word_copy(group_rows[g])}
            for g, f in groups.items()
        },
        "rule_based_examples": pick_rule_examples(pooled_rows),
        "failure_mode_definitions": {
            "repetition_rate": "share of sentences with any repeated word 3-gram in the hypothesis",
            "truncation_rate": "share with hyp/ref word-count ratio < 0.5",
            "untranslated_copy_rate": "share with > 10% of hypothesis subwords copied from the "
            "source and never seen on the train target side",
            "overlong_rate": "share with hyp/ref word-count ratio > 1.5",
            "ref_repetition_rate": "the same repeated-3-gram share computed on the REFERENCES: "
            "repeated 3-grams occur legitimately, so the hypothesis rate is not a degeneration "
            "signal by itself",
            "word_copy_sentence_rate": "HEURISTIC: share of sentences with >= 1 hypothesis word of "
            f"length >= {COPY_MIN_LEN} identical to a source word and absent from the reference",
            "word_copy_words": "number of such words in the group (word_copy_sentences = sentences)",
        },
        "rarity_buckets_e1_e2_e3": {
            "definition": "quintiles of the source's mean subword train frequency, edges from "
            "the pooled E1+E2+E3 sources (model independent); bucket 1 = most common sources",
            "edges_mean_src_freq": edges,
            "buckets": rarity_bucket_chrf(pooled, edges),
        },
    }
    _write(base / "diagnostics.json", out)
    return out


def build_comet_summary(out_root: Path, run: str, variant: str) -> dict[str, Any] | None:
    """Mean COMET-22 with bootstrap CIs per set and per slice (dev slices, E2-synth buckets)."""
    path = out_root / run / variant / "comet.json"
    if not path.is_file():
        return None
    comet = _read(path)
    out: dict[str, Any] = {
        "run": run,
        "variant": variant,
        "model": comet["model"],
        "device": comet["device"],
        "sets": {},
        "slices": {},
    }
    for split in SPLITS:
        scores = comet["per_split_scores"][split]
        out["sets"][split] = bootstrap_mean_ci(scores)
        _, labels = load_split(split)
        slices = [r["slice"] for r in labels]
        if len(set(slices)) > 1:
            out["slices"][split] = slice_means(scores, slices)
    _write(out_root / run / variant / "comet_summary.json", out)
    return out


def build_extras(out_root: Path) -> dict[str, Any]:
    """PREREG §3 'also reported' items the compare JSONs do not hold: the paired test on the
    >80-word source bucket pooled over E1+E2+E3 (Delta = A - B, chrF and BLEU)."""
    extras: dict[str, Any] = {}
    for hyp, run_a, run_b in HYPOTHESES:
        for variant in VARIANTS:
            hs_a, hs_b, refs = [], [], []
            for split in ("e1", "e2", "e3"):
                inputs, labels = load_split(split)
                ref_by_id = {r["id"]: r["reference"] for r in labels}
                pa = _read(out_root / run_a / variant / f"{split}_predictions.json")
                pb = _read(out_root / run_b / variant / f"{split}_predictions.json")
                for r in inputs:
                    if length_bucket_label(len(r["source"].split())) == ">80":
                        hs_a.append(pa[r["id"]])
                        hs_b.append(pb[r["id"]])
                        refs.append(ref_by_id[r["id"]])
            extras[f"{hyp}_{variant}"] = {
                "delta": f"{run_a} - {run_b}",
                "bucket": ">80 source words, pooled E1+E2+E3",
                "n": len(refs),
                **{
                    m: paired_bootstrap(hs_a, hs_b, refs, _official_metric_fn(m), N_BOOTSTRAP, SEED)
                    for m in ("chrf", "bleu")
                },
            }
    _write(out_root / "compare" / "extras.json", extras)
    return extras


def build_sanity(out_root: Path) -> dict[str, Any]:
    """Scorer sanity checks, all recomputed here from files on disk."""
    report: dict[str, Any] = {"pull_records": {}, "official_rescore": {}, "bootstrap_parity": {}}
    for run in RUNS:
        rec = _read(out_root / run / "pull_record.json")
        report["pull_records"][run] = {
            "verified": rec["verified"],
            "private": rec["private"],
            "hf_revision": rec["hf_revision"],
            "pinned_revision_matches": rec["hf_revision"] == PINNED_REVISIONS[run],
            "manifest_sha256": rec["manifest_sha256"],
        }
    max_diff = 0.0
    n_checked = 0
    for run in RUNS:
        for variant in VARIANTS:
            ev = _read(out_root / run / variant / "eval.json")
            for split in SPLITS:
                cli = _read(out_root / run / variant / f"official_{split}.json")
                entry = ev["sets"][split]
                for metric in ("bleu", "chrf"):
                    point = entry[f"official_{metric}_ci"]["point"]
                    max_diff = max(max_diff, abs(point - cli["all"][metric]))
                    n_checked += 1
    report["bootstrap_parity"] = {
        "what": "fast-bootstrap point estimate vs official/score.py CLI report (overall BLEU/chrF)",
        "n_checked": n_checked,
        "max_abs_diff": max_diff,
        "equal": max_diff == 0.0,
    }
    # re-score the dev prediction file of every run, exactly as pulled from the HF source dir,
    # through the repo wrapper (PYTHONUTF8=1) and compare with eval.json
    for run in RUNS:
        for variant in VARIANTS:
            src = out_root / run / "source" / "runs" / run / "predictions" / variant
            with tempfile.TemporaryDirectory() as tmp:
                cli = run_official_scorer_cli(
                    gold_path("dev"), src / "dev_predictions.json", Path(tmp) / "r.json"
                )
            ev = _read(out_root / run / variant / "eval.json")["sets"]["dev"]["official"]
            report["official_rescore"][f"{run}/{variant}"] = {
                "OVERALL": cli["OVERALL"],
                "equals_eval_json": cli["OVERALL"] == ev["OVERALL"] and cli["all"] == ev["all"],
            }
    report["colab_side_dev_numbers"] = (
        "NOT PRESENT: the uploaded runs/<run>/ contain no dev scores (selection.json and tuning/ "
        "use E1+E2 only; dev is decoded after selection), so the dev re-score is checked against "
        "the in-process scorer and the CLI parity record, not a Colab-side figure"
    )
    # nmt.tune picks T on E2 with E1 held fixed, so selection.json's objective is E1 from the
    # unsegmented decode + E2 from the T-segmented decode (verified here, not assumed); the
    # all-segmented objective differs by the E1 sentences that T changes
    sel_check: dict[str, Any] = {}
    for run in RUNS:
        win = _read(out_root / run / "source" / "runs" / run / "selection.json")["winner"]
        e1_off, r1 = _objective_inputs(out_root / run / "seg_off", "e1")
        e1_tuned, _ = _objective_inputs(out_root / run / "seg_tuned", "e1")
        e2_tuned, r2 = _objective_inputs(out_root / run / "seg_tuned", "e2")
        held = _objective_point(e1_off, r1, e2_tuned, r2)["objective"]
        full = _objective_point(e1_tuned, r1, e2_tuned, r2)["objective"]
        sel_check[run] = {
            "selection_json_objective": win["objective"],
            "local_rescore_e1_seg_off_e2_seg_tuned": held,
            "abs_diff": abs(held - win["objective"]),
            "local_rescore_all_seg_tuned": full,
        }
    report["selection_objective_rescore"] = sel_check
    _write(out_root / "sanity.json", report)
    return report


# -------------------------------------------------------------------------------------------
# SUMMARY.md
# -------------------------------------------------------------------------------------------


def _row(
    label: str,
    n: int,
    bleu: dict[str, float],
    chrf: dict[str, float],
    comet: dict[str, float] | None,
    sacre: tuple[float, float] | None,
) -> str:
    c = "-"
    if comet:
        c = f"{comet['point']:.4f} [{comet['ci_low']:.4f}, {comet['ci_high']:.4f}]"
    s = f"{sacre[0]:.2f} / {sacre[1]:.2f}" if sacre else "-"
    return (
        f"| {label} | {n} | {fmt_ci(bleu['point'], bleu['ci_low'], bleu['ci_high'])} | "
        f"{fmt_ci(chrf['point'], chrf['ci_low'], chrf['ci_high'])} | {c} | {s} |"
    )


def model_table(out_root: Path, run: str, variant: str) -> str:
    rel = f"{run}/{variant}" if variant else run  # the baseline has no variant level
    ev = _read(out_root / rel / "eval.json")
    cs_path = out_root / rel / "comet_summary.json"
    cs = _read(cs_path) if cs_path.is_file() else None
    cfg = ev["decoding_config"]
    sets = ev["sets"]
    lines = [
        "| set / slice | n | official BLEU [95% CI] | official chrF [95% CI] | "
        "COMET-22 [95% CI] | sacreBLEU BLEU / chrF |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        e = sets[split]
        sb = (e["sacrebleu"]["bleu"]["score"], e["sacrebleu"]["chrf"]["score"])
        comet = cs["sets"][split] if cs else None
        lines.append(
            _row(SET_LABEL[split], e["n"], e["official_bleu_ci"], e["official_chrf_ci"], comet, sb)
        )
        if split == "dev":
            o = e["overall_ci"]
            lines.append(
                f"| dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen) | {e['n']} | "
                f"{fmt_ci(o['point'], o['ci_low'], o['ci_high'])} | - | - | - |"
            )
        slice_names = list(e["official_ci_by_slice"]["bleu"])
        if split in ("dev", "e2synth"):
            for name in sorted(slice_names):
                comet_s = cs["slices"].get(split, {}).get(name) if cs else None
                lines.append(
                    _row(
                        f"&nbsp;&nbsp;{split}: {name}",
                        e["official_ci_by_slice"]["bleu"][name]["n"],
                        e["official_ci_by_slice"]["bleu"][name],
                        e["official_ci_by_slice"]["chrf"][name],
                        comet_s,
                        None,
                    )
                )
    sig = sets["dev"]["sacrebleu"]
    thr = cfg["segment_threshold"]
    decode = (
        "no model, no decoding (output = source)"
        if cfg["alpha"] is None
        else f"alpha {cfg['alpha']}, beam {cfg['beam_size']}, segmentation "
        f"{'OFF' if thr is None else f'T={thr}'}; checkpoint `{ev['checkpoint']}`"
    )
    head = (
        f"{decode}; sacreBLEU BLEU `{sig['bleu']['signature']}`, chrF `{sig['chrf']['signature']}`"
    )
    src = f"`{rel}/eval.json`" + (f", `{rel}/comet_summary.json`" if cs else "")
    return f"{head}\n\n" + "\n".join(lines) + f"\n\nSource: {src}.\n"


def length_table(out_root: Path, run: str, variant: str) -> str:
    rel = f"{run}/{variant}" if variant else run
    lb = _read(out_root / rel / "eval.json")["length_buckets_e1_e2_e3"]
    lines = ["| source words | n | BLEU [95% CI] | chrF [95% CI] |", "|---|---|---|---|"]
    for label in ("<=10", "11-20", "21-40", "41-80", ">80"):
        if label in lb:
            b = lb[label]
            lines.append(
                f"| {label} | {b['n']} | "
                f"{fmt_ci(b['bleu_ci']['point'], b['bleu_ci']['ci_low'], b['bleu_ci']['ci_high'])} | "
                f"{fmt_ci(b['chrf_ci']['point'], b['chrf_ci']['ci_low'], b['chrf_ci']['ci_high'])} |"
            )
    return "\n".join(lines) + f"\n\nSource: `{rel}/eval.json` (`length_buckets_e1_e2_e3`).\n"


def seg_table(out_root: Path, run: str) -> str:
    off = _read(out_root / run / "seg_off" / "eval.json")
    tuned = _read(out_root / run / "seg_tuned" / "eval.json")
    t = tuned["decoding_config"]["segment_threshold"]
    lines = [
        f"| set | chrF seg OFF | chrF T={t} | delta | BLEU seg OFF | BLEU T={t} | delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        a, b = off["sets"][split]["official"]["all"], tuned["sets"][split]["official"]["all"]
        lines.append(
            f"| {SET_LABEL[split]} | {a['chrf']:.2f} | {b['chrf']:.2f} | {b['chrf'] - a['chrf']:+.2f}"
            f" | {a['bleu']:.2f} | {b['bleu']:.2f} | {b['bleu'] - a['bleu']:+.2f} |"
        )
    return "\n".join(lines) + (
        f"\n\nSource: `{run}/seg_off/eval.json`, `{run}/seg_tuned/eval.json` "
        "(differences are tuned minus off, computed from the two JSONs).\n"
    )


def verdict_block(hyp: str, comp: dict[str, Any], extras: dict[str, Any] | None, label: str) -> str:
    """The pre-registered decision, stated as hypothesis -> delta [95% CI], p -> verdict, with
    every decision-rule component and the 'also reported' items (PREREG §3)."""
    sp = comp["splits"]

    def d(split: str, metric: str = "chrf", sl: str | None = None) -> dict[str, Any]:
        e = sp[split]["overall"] if sl is None else sp[split]["by_slice"][sl]
        return e[metric]

    def line(name: str, c: dict[str, Any]) -> str:
        return (
            f"{name}: delta {fmt_ci(c['delta'], c['ci_low'], c['ci_high'], signed=True)}, "
            f"p {fmt_p(c['p_value'])}"
        )

    e2, syn = d("e2"), d("e2synth")
    sig_e2 = e2["delta"] > 0 and e2["p_value"] < 0.05
    sig_syn = syn["delta"] > 0 and syn["p_value"] < 0.05
    rows = [
        f"- [{'x' if sig_e2 else ' '}] {line('E2 chrF (n=1000)', e2)}",
        f"- [{'x' if sig_syn else ' '}] {line('E2-synth chrF, pooled over 3 buckets (n=300)', syn)}",
    ]
    ok = sig_e2 and sig_syn
    if hyp == "H2":
        e1 = d("e1")
        non_inf = e1["ci_low"] > -0.5
        ok = ok and non_inf
        rows.append(
            f"- [{'x' if non_inf else ' '}] E1 non-inferiority (CI lower bound > -0.5): "
            f"{line('E1 chrF (n=1940)', e1)}; lower bound {e1['ci_low']:+.2f}"
        )
    also = [f"  - {line('dev ' + k + ' chrF', d('dev', 'chrf', k))}" for k in ("seen", "long")]
    also += [
        f"  - {line(f'E2-synth {k} chrF', d('e2synth', 'chrf', k))}"
        for k in sorted(sp["e2synth"]["by_slice"])
    ]
    if extras is not None:
        name80 = f">80-word bucket E1+E2+E3 (n={extras['n']})"
        also.append(f"  - {line(name80 + ' chrF', extras['chrf'])}")
        also.append(f"  - {line(name80 + ' BLEU', extras['bleu'])}")
    also += [
        f"  - {line(name + ' BLEU', d(split, 'bleu'))}"
        for split, name in (("e2", "E2"), ("e2synth", "E2-synth"))
    ]
    obj = comp["selection_objective"]
    also.append(
        f"  - selection objective: delta "
        f"{fmt_ci(obj['delta'], obj['ci_low'], obj['ci_high'], signed=True)}, "
        f"p {fmt_p(obj['p_value'])}"
    )
    first = comp["delta"]
    verdict = "SUPPORTED" if ok else "NOT SUPPORTED"
    hypo = (
        "RoPE improves chrF on long inputs"
        if hyp == "H1"
        else "concat augmentation helps long inputs without hurting seen inputs by more than 0.5 chrF"
    )
    return (
        f"**{hyp} ({label}; delta = {first}, chrF):** {hypo} -> "
        f"E2 {e2['delta']:+.2f} [{e2['ci_low']:+.2f}, {e2['ci_high']:+.2f}] p {fmt_p(e2['p_value'])}"
        f"; E2-synth {syn['delta']:+.2f} [{syn['ci_low']:+.2f}, {syn['ci_high']:+.2f}] "
        f"p {fmt_p(syn['p_value'])} -> **{verdict}**\n\n"
        "Decision-rule components (checked = met):\n\n"
        + "\n".join(rows)
        + "\n\nAlso reported, not decisive:\n\n"
        + "\n".join(also)
        + "\n"
    )


def _examples_md(out_root: Path) -> str:
    lines = [
        "# Auto-selected failure examples",
        "",
        "Selection rule (`nmt.analysis.select_worst_examples`, deterministic, no manual choice): for "
        "each of the 3 official-dev slices (seen, long, unseen_domain) take the 2 sentences with the "
        "lowest sentence chrF (ties keep the dev inputs order); failure type is auto-tagged in this "
        "priority: truncated (hyp/ref < 0.5 words) > repetition (any repeated 3-gram) > "
        "untranslated_copy (> 10% of hypothesis subwords copied from the source) > low_chrf_other. "
        "Models are shown at their tuned decoding config (seg_tuned).",
        "",
        "**Caveat:** 'worst by dev chrF' is dominated by noisy or unrelated references, so these "
        "6 examples often show a reference problem, not a model failure; whether a reference is "
        "unrelated to its source is NOT detected automatically. Each model therefore also gets 2 "
        "rule-based examples (second list), picked deterministically from official dev + E1 + E2 "
        "+ E3: (A) the largest |hyp/ref word ratio - 1| among references with >= 5 words; (B) the "
        "highest hypothesis repeated-3-gram share among sentences whose reference has none (A's "
        "sentence excluded); ties go to the smaller id.",
        "",
    ]
    for run in RUNS:
        ex = _read(out_root / run / "seg_tuned" / "examples.json")
        lines += [f"## {RUN_LABEL[run]}", "", f"Source: `{run}/seg_tuned/examples.json`.", ""]
        for e in ex:
            lines += [
                f"- `{e['id']}` ({e['slice']}, {e['failure_type']}, chrF {e['chrf']:.1f})",
                f"  - source: {e['source']}",
                f"  - reference: {e['reference']}",
                f"  - hypothesis: {e['hypothesis']}",
            ]
        rb = _read(out_root / run / "seg_tuned" / "diagnostics.json")["rule_based_examples"]
        lines += ["", f"Rule-based picks (source: `{run}/seg_tuned/diagnostics.json`):", ""]
        for e in rb:
            lines += [
                f"- `{e['id']}` ({e['slice']}, rule {e['rule']}; hyp/ref ratio "
                f"{e['length_ratio']:.2f}, hyp repeated-3-gram share {e['hyp_repetition_rate']:.2f})",
                f"  - source: {e['source']}",
                f"  - reference: {e['reference']}",
                f"  - hypothesis: {e['hyp']}",
            ]
        lines.append("")
    return "\n".join(lines)


def _gap_table(out_root: Path) -> str:
    lines = [
        "| model | variant | E1-E3 chrF gap | length_dev | repetition | rarity | dialogue | "
        "residual domain | R^2 | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in RUNS:
        for variant in VARIANTS:
            g = _read(out_root / run / variant / "analysis.json")["gap_decomposition"]
            c = g["contributions_chrf"]
            lines.append(
                f"| {run} | {variant} | {g['total_gap_chrf_e1_minus_e3']:.2f} | "
                f"{c['length_dev']:+.2f} | {c['repetition_rate']:+.2f} | {c['rarity']:+.2f} | "
                f"{c['dialogue_punct_density']:+.2f} | {c['residual_domain']:+.2f} | "
                f"{g['r_squared']:.3f} | {g['n']} |"
            )
    return "\n".join(lines) + (
        "\n\nContributions are chrF points of the E1-E3 gap (coefficient x group-mean difference; "
        "residual domain = -coef(domain_e3)). Source: `<run>/<variant>/analysis.json` "
        "(`gap_decomposition`, OLS with HC3 robust SEs).\n"
    )


def _artifact_table(out_root: Path) -> str:
    lines = [
        "| model | variant | dev unseen chrF (orig -> ref-normalised, delta) | "
        "E3 chrF (orig -> ref-normalised, delta) |",
        "|---|---|---|---|",
    ]
    for run in RUNS:
        for variant in VARIANTS:
            a = _read(out_root / run / variant / "analysis.json")["metric_artifact_share"]
            u, e3 = a["dev_unseen_domain"], a["e3"]
            lines.append(
                f"| {run} | {variant} | {u['chrf_original']:.2f} -> {u['chrf_ref_normalized']:.2f} "
                f"({u['delta_chrf']:+.2f}) | {e3['chrf_original']:.2f} -> "
                f"{e3['chrf_ref_normalized']:.2f} ({e3['delta_chrf']:+.2f}) |"
            )
    return (
        "\n".join(lines)
        + "\n\nSource: `<run>/<variant>/analysis.json` (`metric_artifact_share`).\n"
    )


def _failure_table(out_root: Path, variant: str) -> str:
    groups = (
        "dev:seen",
        "dev:long",
        "dev:unseen_domain",
        "e1",
        "e2",
        "e2synth",
        "e3",
    )
    lines = [
        "| model | group | n | repetition (hyp) | repetition (ref) | truncation | "
        "untranslated copy (subword, see caveat) | word-copy heuristic: sentences (words) | "
        "over-long | mean hyp/ref ratio |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in RUNS:
        fm = _read(out_root / run / variant / "diagnostics.json")["failure_modes"]
        for g in groups:
            r = fm[g]
            lines.append(
                f"| {run} | {g} | {r['n']} | {r['repetition_rate']:.3f} | "
                f"{r['ref_repetition_rate']:.3f} | {r['truncation_rate']:.3f}"
                f" | {r['untranslated_copy_rate']:.3f} | {r['word_copy_sentence_rate']:.3f} "
                f"({r['word_copy_sentences']} sent., {r['word_copy_words']} words) | "
                f"{r['overlong_rate']:.3f} | {r['length_ratio_mean']:.3f} |"
            )
    return "\n".join(lines) + (
        f"\n\nShares of sentences (definitions in `<run>/{variant}/diagnostics.json`). Source: "
        f"`<run>/{variant}/diagnostics.json` (`failure_modes`).\n\n"
        "Caveats. (1) *Repetition* counts ANY repeated word 3-gram, including legitimate ones "
        "(references repeat 3-grams too); the reference-side share is printed next to the "
        "hypothesis share and a hypothesis rate at or below it is not a degeneration signal by "
        "itself. (2) *Untranslated copy (subword)* counts only hypothesis subwords that never "
        "occurred on the train target side, so it is 0.000 almost by construction and must NOT be "
        "read as 'no untranslated copies'. The *word-copy heuristic* column measures it directly "
        f"but crudely: hypothesis words of length >= {COPY_MIN_LEN} identical to a source word and "
        "absent from the reference (a heuristic; names the reference spells differently, and "
        "legitimate cognates, are counted).\n"
    )


def _baseline_section(out_root: Path) -> str:
    """The copy-the-source floor (output = source) in the same table format as the models."""
    obj = _read(out_root / BASELINE_RUN / "objective.json")
    return (
        "Output = the French source sentence verbatim, scored by the same code as the runs above "
        "(`scripts/copy_source_baseline.py`); every row of a model table above should be read "
        "against these floors.\n\n"
        + model_table(out_root, BASELINE_RUN, "")
        + "\nLength buckets (E1+E2+E3 pooled):\n\n"
        + length_table(out_root, BASELINE_RUN, "")
        + f"\nSelection objective of the baseline (0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 "
        f"chrF(E1)): {obj['objective']:.4f}. Source: `{BASELINE_RUN}/objective.json`.\n"
    )


def _calibration_table(out_root: Path) -> str:
    """Word-copy heuristic of main/seg_tuned next to the pure-copy ceiling and the reference
    level (what the reference itself keeps from the source), per group."""
    cal = _read(out_root / BASELINE_RUN / "calibration.json")["groups"]
    lines = [
        "| group | n | main: sentences (word share) | copy-source ceiling: sentences (word share) | "
        "reference level: sentences (word share) |",
        "|---|---|---|---|---|",
    ]
    for g, v in cal.items():
        m, b, r = v["model"], v["baseline_ceiling"], v["reference_level"]
        lines.append(
            f"| {g} | {v['n']} | {m['word_copy_sentence_rate']:.3f} ({m['word_copy_word_share']:.3f})"
            f" | {b['word_copy_sentence_rate']:.3f} ({b['word_copy_word_share']:.3f}) | "
            f"{r['ref_sentence_share']:.3f} ({r['ref_word_share']:.3f}) |"
        )
    return "\n".join(lines) + (
        f"\n\nCalibration of the word-copy heuristic: main/seg_tuned vs the same heuristic on the "
        f"copy-the-source baseline (the ceiling) vs the share of reference words of length >= "
        f"{COPY_MIN_LEN} that equal a source word (what the reference itself keeps). Word share = "
        "flagged words / words of length >= 4. How to read it (and its limits) is in "
        f"`{BASELINE_RUN}/README.md`. Source: `{BASELINE_RUN}/calibration.json`.\n"
    )


def _rarity_table(out_root: Path, variant: str) -> str:
    lines = [
        "| model | " + " | ".join(f"bucket {i}" for i in range(1, 6)) + " |",
        "|---|---|---|---|---|---|",
    ]
    n_line = ""
    for run in RUNS:
        rb = _read(out_root / run / variant / "diagnostics.json")["rarity_buckets_e1_e2_e3"]
        lines.append(f"| {run} | " + " | ".join(f"{b['chrf']:.2f}" for b in rb["buckets"]) + " |")
        n_line = ", ".join(str(b["n"]) for b in rb["buckets"])
    return "\n".join(lines) + (
        f"\n\nMean sentence chrF per source-rarity quintile over E1+E2+E3 (bucket 1 = most common "
        f"sources, 5 = rarest; n per bucket {n_line}). Source: `<run>/{variant}/diagnostics.json` "
        "(`rarity_buckets_e1_e2_e3`).\n"
    )


def _provenance(out_root: Path, code_sha: str) -> str:
    lines = [
        "## Provenance",
        "",
        f"- Code: commit `{code_sha}` (HEAD when the artifacts were generated), eval tag of the "
        "Colab decodes `v0.2.4-colab` (adb3c8c781dc70e204e0f48d1a48123b7bcd261f).",
        "- Bootstrap: 1,000 resamples, seed 1234, 95% percentile CIs; paired tests share resample indices.",
        "- HF repo `OWNER/fr-en-transformer-eval` (private), pinned revisions "
        "(each `pull_record.json` has `verified: true`):",
    ]
    for run in RUNS:
        lines.append(f"  - `{run}`: `{PINNED_REVISIONS[run]}`")
    lines += ["", "Files (sha256):", "", "| file | sha256 |", "|---|---|"]
    files: list[Path] = []
    for run in RUNS:
        files.append(out_root / run / "pull_record.json")
        files.append(out_root / run / "index.json")
        files.append(out_root / run / "selection.json")
        for variant in VARIANTS:
            base = out_root / run / variant
            for name in (
                "eval.json",
                "comet.json",
                "comet_summary.json",
                "analysis.json",
                "diagnostics.json",
                "examples.json",
            ):
                if (base / name).is_file():
                    files.append(base / name)
    files += [
        out_root / BASELINE_RUN / n
        for n in ("eval.json", "objective.json", "diagnostics.json", "calibration.json")
    ]
    files += sorted((out_root / "compare").glob("*.json"))
    files += [out_root / "sanity.json"]
    for f in files:
        lines.append(f"| `{f.relative_to(out_root).as_posix()}` | `{sha256_file(f)}` |")
    lines.append("")
    return "\n".join(lines)


def _comet_note(out_root: Path) -> list[str]:
    """A visible statement of COMET coverage: the table cells say '-' only when it is absent."""
    if all((out_root / r / v / "comet_summary.json").is_file() for r in RUNS for v in VARIANTS):
        return []
    partial = out_root / "comet_partial_cpu_INCOMPLETE" / "partial_summary.json"
    extra = (
        f" A partial CPU run is kept, clearly marked, in `{partial.parent.name}/` "
        "(not used in any table here)."
        if partial.is_file()
        else ""
    )
    return [
        "**COMET-22: NOT MEASURED in this report.** The local CPU run was stopped because its "
        "projected time exceeded the 2 h budget; COMET moves to a GPU session." + extra,
        "",
    ]


def build_summary(out_root: Path, code_sha: str) -> str:
    extras = _read(out_root / "compare" / "extras.json")
    sanity = _read(out_root / "sanity.json")
    parts = [
        "# Final evaluation of the 4 runs",
        "",
        "Every number below is printed from a JSON in this directory (named under each table). "
        "Scores are official `score.py` BLEU/chrF (decision metrics) with 95% bootstrap CIs; "
        "sacreBLEU and COMET-22 are reported alongside and never used for a decision "
        "(PREREG §1). Each model is shown at its own tuned decoding config (the config it would "
        "ship with); the segmentation-OFF numbers are in the seg-off section.",
        "",
        *_comet_note(out_root),
        "## Selection check (HF `selection.json`)",
        "",
        "| run | winner | objective | alpha | beam | T |",
        "|---|---|---|---|---|---|",
    ]
    for run in RUNS:
        w = _read(out_root / run / "selection.json")["winner"]
        parts.append(
            f"| {run} | {w['candidate']} | {w['objective']:.4f} | {w['alpha']} | {w['beam']} | "
            f"{w['segment_threshold']} |"
        )
    parts += [
        "",
        "Source: `<run>/selection.json` (verbatim copy of the HF `runs/<run>/selection.json`; "
        "`source/` is gitignored via the repo's `runs/` rule, re-pull with `scripts.eval_local`).",
        "",
    ]
    parts += ["## Per-model results (tuned decoding config)", ""]
    for run in RUNS:
        parts += [f"### {RUN_LABEL[run]}", "", model_table(out_root, run, "seg_tuned"), ""]
        parts += [
            "Length buckets (E1+E2+E3 pooled):",
            "",
            length_table(out_root, run, "seg_tuned"),
            "",
        ]
    parts += ["## Copy-source baseline (floor)", "", _baseline_section(out_root), ""]
    parts += ["## Segmentation off vs tuned", ""]
    for run in RUNS:
        parts += [f"### {RUN_LABEL[run]}", "", seg_table(out_root, run), ""]
    parts += ["## Per-model results with segmentation OFF (primary decoding for H1/H2)", ""]
    for run in RUNS:
        parts += [f"### {RUN_LABEL[run]}", "", model_table(out_root, run, "seg_off"), ""]
    parts += ["## Pre-registered paired tests (PREREG §3)", ""]
    for hyp, run_a, run_b in HYPOTHESES:
        for variant, label in (
            ("seg_off", "PRIMARY, segmentation off"),
            ("seg_tuned", "secondary, tuned"),
        ):
            comp = _read(out_root / "compare" / f"{hyp}_{run_a}_vs_{run_b}_{variant}.json")
            parts += [
                verdict_block(hyp, comp, extras[f"{hyp}_{variant}"], label),
                f"Source: `compare/{hyp}_{run_a}_vs_{run_b}_{variant}.json`, `compare/extras.json`.",
                "",
            ]
    parts += [
        "## Analysis",
        "",
        "### Gap decomposition (OLS over E1 union E3)",
        "",
        _gap_table(out_root),
        "### Reference-normalisation artifact",
        "",
        _artifact_table(out_root),
        "### Failure-mode rates per slice (tuned config)",
        "",
        _failure_table(out_root, "seg_tuned"),
        "### Word-copy heuristic calibration (main, tuned config)",
        "",
        _calibration_table(out_root),
        "### Chrf by source-rarity bucket (tuned config)",
        "",
        _rarity_table(out_root, "seg_tuned"),
        "Length buckets are in each model's table above; failure examples are in `EXAMPLES.md`.",
        "",
        "## Sanity checks",
        "",
        f"- Pull records verified for all 4 runs: "
        f"{all(v['verified'] and v['pinned_revision_matches'] for v in sanity['pull_records'].values())}.",
        f"- Fast-bootstrap point estimates vs official CLI numbers: "
        f"{sanity['bootstrap_parity']['n_checked']} comparisons, max abs diff "
        f"{sanity['bootstrap_parity']['max_abs_diff']}.",
        f"- Dev prediction file re-scored through the wrapper equals eval.json for "
        f"{sum(v['equals_eval_json'] for v in sanity['official_rescore'].values())}/"
        f"{len(sanity['official_rescore'])} (run, variant) pairs.",
        "- Selection objective re-scored locally (E1 unsegmented + E2 at tuned T, as nmt.tune scores it) vs `selection.json`, max abs diff "
        f"{max(v['abs_diff'] for v in sanity['selection_objective_rescore'].values()):.2e}.",
        f"- {sanity['colab_side_dev_numbers']}.",
        "",
        "Source: `sanity.json`.",
        "",
        _provenance(out_root, code_sha),
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Derived artifacts + SUMMARY.md for reports/final.")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--reuse-sanity",
        action="store_true",
        help="keep the committed index.json, selection.json copies and sanity.json instead of "
        "refreshing them (they need the gitignored HF `source/` pulls)",
    )
    args = p.parse_args(argv)
    root: Path = args.out_root
    if not args.reuse_sanity:
        print(f"normalized line endings of {normalize_and_reindex(root)} files")
    for run in RUNS if not args.reuse_sanity else ():  # winner record: tiny, `source/` gitignored
        shutil.copyfile(
            root / run / "source" / "runs" / run / "selection.json", root / run / "selection.json"
        )
    res = _resources()
    for run in RUNS:
        for variant in VARIANTS:
            build_diagnostics(root, run, variant, res)
            build_comet_summary(root, run, variant)
    build_extras(root)
    if not args.reuse_sanity:
        build_sanity(root)
    code_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    (root / "SUMMARY.md").write_text(build_summary(root, code_sha), encoding="utf-8", newline="\n")
    (root / "EXAMPLES.md").write_text(_examples_md(root), encoding="utf-8", newline="\n")
    print(f"build_final_report: wrote SUMMARY.md and EXAMPLES.md under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
