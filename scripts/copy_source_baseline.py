from __future__ import annotations

# ruff: noqa: E501
# (E501: the README text lines are kept as readable single lines)
# Copy-the-source baseline: the "translation" is the French source sentence verbatim. It is the
# floor of every results table (what a system that does nothing scores) and the calibration
# ceiling of the word-level untranslated-copy heuristic (`scripts.build_final_report.word_copies`):
# a hypothesis that copies everything cannot have a higher copy rate than this.
#
# Scores through the same code the 4 runs used (`nmt.evaluate.score_split_predictions`, official
# `score.py` via `run_official_scorer`, 1,000 resamples, seed 1234), so `eval.json` has the same
# schema as `reports/final/<run>/seg_tuned/eval.json`. Never edits official/score.py, never
# decodes, never touches the test set (no test-set file is written).
#
# Writes under <out-dir> (default reports/final/baseline_copy_source):
#   <split>_predictions.json, official_<split>.json, eval.json   (the models' schema)
#   objective.json    the selection-objective value (0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1))
#   diagnostics.json  failure-mode rates (same functions as the models)
#   calibration.json  word-copy heuristic: model vs baseline ceiling vs reference level
#   README.md         what this is, the numbers, the reading of the heuristic, provenance footer
#
# CLI: `python -m scripts.copy_source_baseline` (PYTHONUTF8=1 is set for the scorer by the wrapper)
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from nmt.analysis import (
    DEFAULT_FREQ_SRC_PATH,
    DEFAULT_FREQ_TGT_PATH,
    DEFAULT_TOKENIZER_PATH,
    _rows_with_predictions,
    _words,
)
from nmt.compare import _objective_point
from nmt.evaluate import (
    OFFICIAL_SCORE_PY,
    REPO_ROOT,
    length_bucket_view,
    load_split,
    score_split_predictions,
)
from scripts.build_final_report import (
    COPY_MIN_LEN,
    N_BOOTSTRAP,
    SEED,
    _read,
    _resources,
    build_diagnostics,
    word_copies,
)
from scripts.eval_local import SPLITS, _write_json, official_cli_parity

CRLF, LF = bytes([13, 10]), bytes([10])  # official/score.py writes CRLF on Windows; git stores LF
RUN = "baseline_copy_source"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "final" / RUN
MODEL_DIR = REPO_ROOT / "reports" / "final" / "main" / "seg_tuned"
LABEL = "copy-source baseline (floor)"


def copy_predictions(inputs: list[dict[str, Any]]) -> dict[str, str]:
    """id -> the source sentence verbatim (the baseline's whole 'model')."""
    return {r["id"]: r["source"] for r in inputs}


def _words_ge_min(text: str) -> list[str]:
    return [w for w in _words(text) if len(w) >= COPY_MIN_LEN]


def reference_copy_level(rows: list[dict[str, str]]) -> dict[str, Any]:
    """The 'legitimate copy' level of a group: reference words of length >= COPY_MIN_LEN that
    equal a source word (case-insensitive), i.e. names, numbers and cognates the REFERENCE itself
    keeps. `rows`: dicts with `source` and `reference`. Reports the word share and the share of
    sentences with at least one such word (the latter is the unit of the heuristic's sentence
    rate). Empty group -> None shares (not 0)."""
    n_words = n_kept = n_sent = 0
    for r in rows:
        src = {w.lower() for w in _words(r["source"])}
        ref_words = _words_ge_min(r["reference"])
        kept = [w for w in ref_words if w.lower() in src]
        n_words += len(ref_words)
        n_kept += len(kept)
        n_sent += 1 if kept else 0
    n = len(rows)
    return {
        "n": n,
        "ref_words_ge_min": n_words,
        "ref_words_equal_to_source_word": n_kept,
        "ref_word_share": n_kept / n_words if n_words else None,
        "ref_sentences_with_any": n_sent,
        "ref_sentence_share": n_sent / n if n else None,
    }


def hyp_copy_level(rows: list[dict[str, str]]) -> dict[str, Any]:
    """The word-copy heuristic (`word_copies`) over a group of rows with `hyp`, recomputed from
    predictions, plus its denominator (hypothesis words of length >= COPY_MIN_LEN)."""
    copies = [word_copies(r["hyp"], r["source"], r["reference"]) for r in rows]
    n_words = sum(len(_words_ge_min(r["hyp"])) for r in rows)
    n_copy_words = sum(len(c) for c in copies)
    n = len(rows)
    return {
        "word_copy_sentences": sum(1 for c in copies if c),
        "word_copy_sentence_rate": sum(1 for c in copies if c) / n if n else None,
        "word_copy_words": n_copy_words,
        "hyp_words_ge_min": n_words,
        "word_copy_word_share": n_copy_words / n_words if n_words else None,
    }


def _group_rows(preds_by_split: dict[str, dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Rows (id, source, reference, hyp, slice) per set and per slice, named like the groups of
    `diagnostics.json` ('dev', 'dev:seen', ..., 'e1', 'e2', 'e2synth', 'e2synth:<bucket>', 'e3')."""
    groups: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        rows = _rows_with_predictions(split, preds_by_split[split])
        groups[split] = rows
        names = sorted({r["slice"] for r in rows})
        if len(names) > 1:
            for name in names:
                groups[f"{split}:{name}"] = [r for r in rows if r["slice"] == name]
    return groups


def build_calibration(
    baseline_preds: dict[str, dict[str, str]],
    model_preds: dict[str, dict[str, str]],
    baseline_diag: dict[str, Any],
    model_diag: dict[str, Any],
) -> dict[str, Any]:
    """Per group: the model's word-copy level, the baseline's (ceiling) and the reference level.
    The model/baseline sentence counts are recomputed from predictions AND cross-checked against
    the `diagnostics.json` values they were produced with (a mismatch raises, never averages)."""
    base_groups, model_groups = _group_rows(baseline_preds), _group_rows(model_preds)
    out: dict[str, Any] = {}
    for g, rows in base_groups.items():
        base = hyp_copy_level(rows)
        model = hyp_copy_level(model_groups[g])
        for name, level, diag in (("baseline", base, baseline_diag), ("model", model, model_diag)):
            d = diag["failure_modes"][g]
            if (level["word_copy_sentences"], level["word_copy_words"]) != (
                d["word_copy_sentences"],
                d["word_copy_words"],
            ):
                raise ValueError(f"{name} {g}: recomputed word-copy counts differ from diagnostics")
        out[g] = {
            "n": len(rows),
            "model": {
                **model,
                "length_ratio_mean": model_diag["failure_modes"][g]["length_ratio_mean"],
                "subword_untranslated_copy_rate": model_diag["failure_modes"][g][
                    "untranslated_copy_rate"
                ],
            },
            "baseline_ceiling": {
                **base,
                "length_ratio_mean": baseline_diag["failure_modes"][g]["length_ratio_mean"],
                "subword_untranslated_copy_rate": baseline_diag["failure_modes"][g][
                    "untranslated_copy_rate"
                ],
            },
            "reference_level": reference_copy_level(rows),
        }
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> dict[str, str]:
    """sha256 of every input the baseline reads (dev/E-set inputs and labels, tokenizer + the
    frequency tables the heuristics use, the official scorer)."""
    files = [
        *(gold.parent / n for gold in _split_gold() for n in ("inputs.jsonl", "labels.jsonl")),
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_FREQ_SRC_PATH,
        DEFAULT_FREQ_TGT_PATH,
        OFFICIAL_SCORE_PY,
    ]
    return {p.relative_to(REPO_ROOT).as_posix(): _sha256(p) for p in files}


def _split_gold() -> list[Path]:
    return [
        REPO_ROOT / "data" / ("dev" if s == "dev" else f"eval/{s}") / "labels.jsonl" for s in SPLITS
    ]


def _fmt(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _span(calibration: dict[str, Any], who: str, key: str) -> str:
    vals = [v[who][key] for v in calibration.values() if v[who][key] is not None]
    lo, hi = min(vals), max(vals)
    return f"{lo:.3f}" if lo == hi else f"{lo:.3f} to {hi:.3f}"


def reading(calibration: dict[str, Any]) -> str:
    """What the heuristic can and cannot tell, with the ranges printed from `calibration`
    (the prose is fixed, every number in it is computed)."""
    e3 = calibration["e3"]["model"]["word_copy_word_share"]
    return (
        "## What the heuristic can and cannot tell\n\n"
        "On a pure copy of the source the word-copy heuristic flags "
        f"{_span(calibration, 'baseline_ceiling', 'word_copy_word_share')} of the length >= 4 "
        "words across groups (the ceiling), while main/seg_tuned has "
        f"{_span(calibration, 'model', 'word_copy_word_share')} flagged, so the word share "
        "separates copying from translating by a wide margin in every group. "
        "The sentence-level rate is not comparable across sets: the model's own sentence rate "
        f"ranges {_span(calibration, 'model', 'word_copy_sentence_rate')} across groups and is "
        "highest on the long E2-synth inputs while its word share stays low, so the word share is "
        "the figure to read (the sentence rate is consistent with tracking sentence length; that "
        "was not tested directly). "
        "The flagged words are words the reference does not use, so the heuristic cannot tell an "
        "untranslated word from a name spelled differently in the reference or a cognate chosen "
        "where the reference paraphrased (E3, the books proxy, has the highest model share, "
        f"{e3:.3f}; whether those words are names, cognates or true copies was not examined), and "
        "it misses copied words shorter than 4 characters or that also occur in the reference. "
        "The reference level (words the reference itself keeps from the source: "
        f"{_span(calibration, 'reference_level', 'ref_word_share')} of its length >= 4 words) "
        "shows such shared words are common and legitimate, but they are excluded from the "
        "heuristic by construction, so it is context for the ceiling, not a value to subtract. "
        "The older subword-based `untranslated_copy_rate` is a weak detector even on a pure copy "
        f"({_span(calibration, 'baseline_ceiling', 'subword_untranslated_copy_rate')} across "
        "groups for the baseline, against "
        f"{_span(calibration, 'model', 'subword_untranslated_copy_rate')} for the model), so its "
        "0.000 for the model is not evidence of the absence of copying."
    )


def build_readme(
    ev: dict[str, Any],
    objective: dict[str, Any],
    calibration: dict[str, Any],
    code_sha: str,
    hashes: dict[str, str],
    out_dir: Path,
) -> str:
    """README.md: what the baseline is, headline numbers (printed from eval.json / objective.json
    / calibration.json), the reading of the heuristic and the provenance footer."""
    lines = [
        f"# {LABEL}",
        "",
        "Output = the French source sentence verbatim, for every sentence of official dev, E1, E2,",
        "E2-synth and E3. No model, no decoding, no tuning; scored by the same code as the 4 runs",
        "(`nmt.evaluate.score_split_predictions`, official `score.py` through",
        "`nmt.evaluate.run_official_scorer`, 1,000 resamples, seed 1234). No test-set file is written.",
        "",
        "## Scores (official BLEU / chrF, 95% bootstrap CI)",
        "",
        "| set / slice | n | BLEU [95% CI] | chrF [95% CI] |",
        "|---|---|---|---|",
    ]
    for split in SPLITS:
        e = ev["sets"][split]
        b, c = e["official_bleu_ci"], e["official_chrf_ci"]
        lines.append(
            f"| {split} | {e['n']} | {b['point']:.2f} [{b['ci_low']:.2f}, {b['ci_high']:.2f}] | "
            f"{c['point']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}] |"
        )
        if split in ("dev", "e2synth"):
            for name in sorted(e["official_ci_by_slice"]["bleu"]):
                b = e["official_ci_by_slice"]["bleu"][name]
                c = e["official_ci_by_slice"]["chrf"][name]
                lines.append(
                    f"| &nbsp;&nbsp;{split}: {name} | {b['n']} | {b['point']:.2f} "
                    f"[{b['ci_low']:.2f}, {b['ci_high']:.2f}] | {c['point']:.2f} "
                    f"[{c['ci_low']:.2f}, {c['ci_high']:.2f}] |"
                )
    o = ev["sets"]["dev"]["overall_ci"]
    lines += [
        "",
        f"dev OVERALL (official 0.4 BLEU + 0.4 chrF + 0.2 chrF unseen): {o['point']:.2f} "
        f"[{o['ci_low']:.2f}, {o['ci_high']:.2f}].",
        f"Selection objective (0.4 BLEU(E1+E2) + 0.4 chrF(E1+E2) + 0.2 chrF(E1)): "
        f"{objective['objective']:.4f} (`objective.json`).",
        "Source: `eval.json`, `objective.json`.",
        "",
        "## Word-copy heuristic calibration",
        "",
        f"Heuristic (`scripts.build_final_report.word_copies`): hypothesis words of length >= "
        f"{COPY_MIN_LEN} equal to a source word and absent from the reference. Reference level: "
        f"share of reference words of length >= {COPY_MIN_LEN} that equal a source word (names, "
        "numbers, cognates the reference itself keeps). Model = main/seg_tuned. Source: "
        "`calibration.json`.",
        "",
        "| group | n | model: sentences (words) | baseline ceiling: sentences (words) | "
        "reference level: sentences (words) |",
        "|---|---|---|---|---|",
    ]
    for g, v in calibration.items():
        m, bc, r = v["model"], v["baseline_ceiling"], v["reference_level"]
        lines.append(
            f"| {g} | {v['n']} | {_fmt(m['word_copy_sentence_rate'])} "
            f"({_fmt(m['word_copy_word_share'])}) | {_fmt(bc['word_copy_sentence_rate'])} "
            f"({_fmt(bc['word_copy_word_share'])}) | {_fmt(r['ref_sentence_share'])} "
            f"({_fmt(r['ref_word_share'])}) |"
        )
    lines += [
        "",
        "Each cell is the share of sentences with at least one flagged word, and in parentheses the",
        "share of length >= 4 words flagged (hypothesis words for model and baseline, reference",
        "words for the reference level).",
        "",
        reading(calibration),
        "",
        "## Provenance",
        "",
        f"- Code: commit `{code_sha}` (HEAD when generated); `python -m scripts.copy_source_baseline`.",
        "- Input sha256 (LF, no CR in any generated text file):",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | `{v}` |" for k, v in hashes.items()]
    lines += [
        f"| `{MODEL_DIR.relative_to(REPO_ROOT).as_posix()}/diagnostics.json` | "
        f"`{_sha256(MODEL_DIR / 'diagnostics.json')}` |",
        "",
        "Generated files (sha256):",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for name in ("eval.json", "objective.json", "diagnostics.json", "calibration.json"):
        lines.append(f"| `{name}` | `{_sha256(out_dir / name)}` |")
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Path, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    """Produce every baseline artifact under `out_dir` and return the calibration dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "run": RUN,
        "checkpoint": "none (copy-source baseline: output = source)",
        "variant": "baseline",
        "decoding_config": {
            "beam_size": None,
            "alpha": None,
            "segment_threshold": None,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": seed,
        },
        "provenance": {"baseline": "output = French source sentence verbatim; no model"},
        "sets": {},
    }
    preds: dict[str, dict[str, str]] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    all_pred: dict[str, str] = {}
    for split in SPLITS:
        inputs, labels = load_split(split)
        pred = preds[split] = copy_predictions(inputs)
        pred_path = out_dir / f"{split}_predictions.json"
        pred_path.write_bytes(
            (json.dumps(pred, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        entry = score_split_predictions(split, inputs, labels, pred, n_bootstrap, seed)
        entry["official_cli"] = official_cli_parity(split, pred_path, out_dir, entry)
        # official/score.py writes with the platform newline (CRLF on Windows); git stores LF
        report = out_dir / f"official_{split}.json"
        report.write_bytes(report.read_bytes().replace(CRLF, LF))
        result["sets"][split] = entry
        ref_by_id = {r["id"]: r["reference"] for r in labels}
        all_pred.update(pred)
        rows_by_split[split] = [
            {"id": r["id"], "source": r["source"], "reference": ref_by_id[r["id"]]} for r in inputs
        ]
    combined = [row for s in ("e1", "e2", "e3") for row in rows_by_split[s]]
    result["length_buckets_e1_e2_e3"] = length_bucket_view(combined, all_pred, n_bootstrap, seed)
    _write_json(out_dir / "eval.json", result)

    refs = {s: [r["reference"] for r in rows_by_split[s]] for s in ("e1", "e2")}
    hyps = {s: [preds[s][r["id"]] for r in rows_by_split[s]] for s in ("e1", "e2")}
    objective = _objective_point(hyps["e1"], refs["e1"], hyps["e2"], refs["e2"])
    objective["formula"] = "0.4*BLEU(E1+E2) + 0.4*chrF(E1+E2) + 0.2*chrF(E1), official score.py"
    objective["n_e1"], objective["n_e2"] = len(refs["e1"]), len(refs["e2"])
    sanity = _read(REPO_ROOT / "reports" / "final" / "sanity.json")["selection_objective_rescore"]
    objective["main_seg_tuned_local_rescore_all_seg_tuned"] = sanity["main"][
        "local_rescore_all_seg_tuned"
    ]
    _write_json(out_dir / "objective.json", objective)

    base_diag = build_diagnostics(out_dir.parent, RUN, "baseline", _resources(), base_dir=out_dir)
    model_preds = {s: _read(MODEL_DIR / f"{s}_predictions.json") for s in SPLITS}
    model_diag = _read(MODEL_DIR / "diagnostics.json")
    calibration = build_calibration(preds, model_preds, base_diag, model_diag)
    _write_json(
        out_dir / "calibration.json",
        {
            "what": "word-copy heuristic: main/seg_tuned vs copy-source baseline (ceiling) vs "
            "reference level",
            "min_word_len": COPY_MIN_LEN,
            "groups": calibration,
        },
    )
    code_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    readme = build_readme(result, objective, calibration, code_sha, input_hashes(), out_dir)
    (out_dir / "README.md").write_bytes(readme.encode("utf-8"))
    return calibration


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Copy-the-source baseline artifacts.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)
    run(args.out_dir)
    print(f"copy_source_baseline: wrote artifacts under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
