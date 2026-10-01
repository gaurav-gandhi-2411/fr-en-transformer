from __future__ import annotations

# Evaluation: runs the vendored official/score.py exactly as shipped (subprocess CLI, primary
# metric), re-derives the same numbers in-process via importlib (for per-sentence chrF and
# bootstrap resampling, without ever modifying official/score.py), sacreBLEU BLEU/chrF/chrF++
# with signatures, optional COMET-22 (eval-only, isolated env, never used for selection),
# bootstrap 95% CIs and paired bootstrap A/B comparisons, reported by slice / length bucket /
# E-set. Spec §8.
import importlib.util
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import sacrebleu
import sentencepiece as spm
import torch

from nmt.decode import greedy_decode
from nmt.translate import Translator, _detok, _pad_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCORE_PY = REPO_ROOT / "official" / "score.py"

LENGTH_BUCKET_LABELS = ("<=10", "11-20", "21-40", "41-80", ">80")

_official_module_cache: ModuleType | None = None


def load_official_module() -> ModuleType:
    """Import `official/score.py`'s functions in-process via `importlib`, without ever editing
    the vendored file (spec §2/§8: run it "exactly as shipped"; sha256-checked separately in
    `tests/test_official_scorer.py`). Cached after first import.
    """
    global _official_module_cache
    if _official_module_cache is None:
        spec = importlib.util.spec_from_file_location("official_score", OFFICIAL_SCORE_PY)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _official_module_cache = module
    return _official_module_cache


def run_official_scorer_cli(gold_path: Path, pred_path: Path, out_path: Path) -> dict[str, Any]:
    """Run `official/score.py` exactly as shipped via `subprocess` (the primary metric, spec §8),
    parsing the `--out` JSON report it writes."""
    subprocess.run(
        [
            sys.executable,
            str(OFFICIAL_SCORE_PY),
            "--gold",
            str(gold_path),
            "--pred",
            str(pred_path),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def compute_official_metrics(pred: dict[str, str], gold_rows: list[dict]) -> dict[str, Any]:
    """In-process re-derivation of `official/score.py::main`'s report (same `{"all", "OVERALL",
    "by_slice"}` shape), calling the *same* imported functions on the *same* inputs the CLI would
    see -- `tests/test_evaluate.py` checks this equals `run_official_scorer_cli`'s output exactly
    on a fixed prediction set (spec §12).
    """
    module = load_official_module()
    gold = {r["id"]: r for r in gold_rows}
    ids = list(gold)
    hyps = [pred.get(i, "") for i in ids]
    refs = [gold[i].get("reference", "") for i in ids]
    by: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for i in ids:
        h, r = pred.get(i, ""), gold[i].get("reference", "")
        sl = gold[i].get("slice", "unspecified")
        by[sl][0].append(h)
        by[sl][1].append(r)
    all_scores = module.score_slice(hyps, refs)
    by_slice = {k: module.score_slice(h, r) for k, (h, r) in by.items()}
    unseen = by_slice.get("unseen_domain")
    unseen_chrf = unseen["chrf"] if unseen else all_scores["chrf"]
    overall = 0.40 * all_scores["bleu"] + 0.40 * all_scores["chrf"] + 0.20 * unseen_chrf
    return {"all": all_scores, "OVERALL": overall, "by_slice": by_slice}


# -------------------------------------------------------------------------------------------
# sacreBLEU
# -------------------------------------------------------------------------------------------


def sacrebleu_metrics(hyps: list[str], refs: list[str]) -> dict[str, Any]:
    """sacreBLEU BLEU, chrF and chrF++ with their reproducibility signatures (spec §8)."""
    bleu = sacrebleu.BLEU()
    chrf = sacrebleu.CHRF()
    chrfpp = sacrebleu.CHRF(word_order=2)
    return {
        "bleu": {
            "score": bleu.corpus_score(hyps, [refs]).score,
            "signature": bleu.get_signature().format(),
        },
        "chrf": {
            "score": chrf.corpus_score(hyps, [refs]).score,
            "signature": chrf.get_signature().format(),
        },
        "chrf++": {
            "score": chrfpp.corpus_score(hyps, [refs]).score,
            "signature": chrfpp.get_signature().format(),
        },
    }


# -------------------------------------------------------------------------------------------
# Bootstrap statistics
# -------------------------------------------------------------------------------------------


def _official_metric_fn(metric: str) -> Callable[[list[str], list[str]], float]:
    module = load_official_module()
    if metric == "bleu":
        return module.corpus_bleu
    if metric == "chrf":
        return lambda h, r: (
            sum(module.chrf_sentence(a, b) for a, b in zip(h, r, strict=True)) / len(r)
            if r
            else 0.0
        )
    raise ValueError(f"unknown metric: {metric!r}")


def bootstrap_ci(
    hyps: list[str],
    refs: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Bootstrap 95% CI (spec §8: 1000 resamples, seeded) for a corpus-level metric. Resamples
    *sentence indices* (paired hyp/ref) with replacement and recomputes the corpus metric on each
    resample -- correct for non-additive corpus metrics like BLEU, unlike averaging per-sentence
    values."""
    rng = random.Random(seed)
    n = len(refs)
    point = metric_fn(hyps, refs)
    if n == 0:
        return {
            "point": point,
            "ci_low": point,
            "ci_high": point,
            "n_resamples": n_resamples,
            "n": 0,
        }
    samples = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(metric_fn([hyps[i] for i in idx], [refs[i] for i in idx]))
    samples.sort()
    lo = samples[int(0.025 * n_resamples)]
    hi = samples[min(n_resamples - 1, int(0.975 * n_resamples))]
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_resamples": n_resamples, "n": n}


def bootstrap_ci_official(
    hyps: list[str], refs: list[str], metric: str, n_resamples: int = 1000, seed: int = 1234
) -> dict[str, Any]:
    """`bootstrap_ci` specialized to the official scorer's own BLEU/chrF implementations."""
    return bootstrap_ci(hyps, refs, _official_metric_fn(metric), n_resamples, seed)


def bootstrap_official_overall(
    pred: dict[str, str], gold_rows: list[dict], n_resamples: int = 1000, seed: int = 1234
) -> dict[str, Any]:
    """Bootstrap 95% CI for the official OVERALL formula (0.4*BLEU + 0.4*chrF + 0.2*chrF(unseen))
    -- each resample draws sentence indices jointly (paired across slices) and recomputes OVERALL
    on that resample, so the CI reflects the actual composite metric, not three independent CIs.
    """
    module = load_official_module()
    ids = [r["id"] for r in gold_rows]
    n = len(ids)
    hyps_all = [pred.get(i, "") for i in ids]
    refs_all = [r.get("reference", "") for r in gold_rows]
    slices = [r.get("slice", "unspecified") for r in gold_rows]

    def overall_for(idxs: list[int]) -> float:
        h = [hyps_all[i] for i in idxs]
        r = [refs_all[i] for i in idxs]
        s = [slices[i] for i in idxs]
        all_scores = module.score_slice(h, r)
        ud_h = [h[i] for i in range(len(idxs)) if s[i] == "unseen_domain"]
        ud_r = [r[i] for i in range(len(idxs)) if s[i] == "unseen_domain"]
        ud_chrf = module.score_slice(ud_h, ud_r)["chrf"] if ud_h else all_scores["chrf"]
        return 0.40 * all_scores["bleu"] + 0.40 * all_scores["chrf"] + 0.20 * ud_chrf

    rng = random.Random(seed)
    point = overall_for(list(range(n)))
    if n == 0:
        return {
            "point": point,
            "ci_low": point,
            "ci_high": point,
            "n_resamples": n_resamples,
            "n": 0,
        }
    samples = sorted(overall_for([rng.randrange(n) for _ in range(n)]) for _ in range(n_resamples))
    lo = samples[int(0.025 * n_resamples)]
    hi = samples[min(n_resamples - 1, int(0.975 * n_resamples))]
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_resamples": n_resamples, "n": n}


def paired_bootstrap(
    hyps_a: list[str],
    hyps_b: list[str],
    refs: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Paired bootstrap resampling (Koehn 2004) for an A/B comparison: Delta = metric(A) -
    metric(B), a 95% CI on Delta, and a one-sided p-value (the fraction of resamples where B's
    score meets or exceeds A's -- evidence *against* "A is better"). Used for ablations and
    decoding-option comparisons (spec §8/§9).
    """
    rng = random.Random(seed)
    n = len(refs)
    point_a = metric_fn(hyps_a, refs)
    point_b = metric_fn(hyps_b, refs)
    delta_point = point_a - point_b
    if n == 0:
        return {
            "delta": delta_point,
            "ci_low": delta_point,
            "ci_high": delta_point,
            "p_value": 1.0,
            "n_resamples": n_resamples,
        }
    deltas = []
    count_b_ge_a = 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ha = [hyps_a[i] for i in idx]
        hb = [hyps_b[i] for i in idx]
        r = [refs[i] for i in idx]
        d = metric_fn(ha, r) - metric_fn(hb, r)
        deltas.append(d)
        if d <= 0:
            count_b_ge_a += 1
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[min(n_resamples - 1, int(0.975 * n_resamples))]
    p_value = count_b_ge_a / n_resamples
    return {
        "delta": delta_point,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "n_resamples": n_resamples,
    }


# -------------------------------------------------------------------------------------------
# Views
# -------------------------------------------------------------------------------------------


def length_bucket_label(n_words: int) -> str:
    """Source-length-in-words bucket label (spec §8: <=10, 11-20, 21-40, 41-80, >80)."""
    if n_words <= 10:
        return "<=10"
    if n_words <= 20:
        return "11-20"
    if n_words <= 40:
        return "21-40"
    if n_words <= 80:
        return "41-80"
    return ">80"


def length_bucket_view(rows: list[dict[str, str]], pred: dict[str, str]) -> dict[str, Any]:
    """`rows`: [{id, source, reference}, ...] pooled across E1+E2+E3 (spec §8). Returns official
    BLEU/chrF per non-empty length bucket."""
    module = load_official_module()
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        label = length_bucket_label(len(row["source"].split()))
        buckets[label].append((pred.get(row["id"], ""), row["reference"]))
    return {
        label: module.score_slice([p[0] for p in pairs], [p[1] for p in pairs])
        for label, pairs in buckets.items()
        if pairs and label in LENGTH_BUCKET_LABELS
    }


# -------------------------------------------------------------------------------------------
# COMET (eval-only; isolated env via envs/comet, spec §8)
# -------------------------------------------------------------------------------------------

COMET_ENV_DIR = REPO_ROOT / "envs" / "comet"
COMET_SCRIPT = COMET_ENV_DIR / "score_comet.py"


def run_comet(
    triples: list[dict[str, str]], out_path: Path, timeout_seconds: float = 3600.0
) -> dict[str, Any]:
    """Score `triples` (each `{"src", "mt", "ref"}`) with COMET-22 (`Unbabel/wmt22-comet-da`) by
    shelling out to the isolated `envs/comet` uv project (`unbabel-comet` does not co-resolve
    with this repo's pinned torch/numpy -- see pyproject.toml). Eval-only: never used for
    checkpoint/decoding selection (spec §8, §15). Raises `RuntimeError` with the subprocess's
    stderr on failure -- never silently fabricates a score.
    """
    in_path = out_path.with_suffix(".in.json")
    in_path.write_text(json.dumps(triples), encoding="utf-8")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(COMET_ENV_DIR),
            "python",
            str(COMET_SCRIPT),
            "--in",
            str(in_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"COMET scoring failed (exit {result.returncode}):\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------------------------------
# Top-level orchestration
# -------------------------------------------------------------------------------------------


@dataclass
class EvalRunConfig:
    """Decoding + statistics config for one `run_evaluation` call. Recorded verbatim in eval.json
    ("decoding config", spec §8)."""

    beam_size: int = 5
    alpha: float = 0.6
    segment_threshold: int | None = None
    batch_size: int = 16
    n_bootstrap: int = 1000
    bootstrap_seed: int = 1234
    comet: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns `(inputs_rows, labels_rows)` for `name` in {"e1", "e2", "e3", "dev"}."""
    if name in ("e1", "e2", "e3"):
        base = REPO_ROOT / "data" / "eval" / name
    elif name == "dev":
        base = REPO_ROOT / "data" / "dev"
    else:
        raise ValueError(f"unknown split: {name!r}")
    return _read_jsonl(base / "inputs.jsonl"), _read_jsonl(base / "labels.jsonl")


def run_evaluation(
    translator: Translator,
    run_name: str,
    ckpt_name: str,
    decode_cfg: EvalRunConfig,
    out_dir: Path | None = None,
    splits: Sequence[str] = ("dev", "e1", "e2", "e3"),
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate every `splits` entry with `translator`, score it (official + sacreBLEU +
    bootstrap CIs), build the length-bucket view over E1+E2+E3, optionally run COMET, and write
    `reports/<run_name>/<ckpt_name>/eval.json` + each split's predictions (spec §8 output shape).
    """
    out_dir = out_dir or (REPO_ROOT / "reports" / run_name / ckpt_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "run": run_name,
        "checkpoint": ckpt_name,
        "decoding_config": asdict(decode_cfg),
        "provenance": provenance or {},
        "timings_seconds": {},
        "sets": {},
    }

    all_pred: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}

    for split in splits:
        t0 = time.monotonic()
        inputs, labels = load_split(split)
        label_by_id = {r["id"]: r for r in labels}
        ids = [r["id"] for r in inputs]
        sources = [r["source"] for r in inputs]

        stats_before = (
            translator.stats.n_beam,
            translator.stats.n_greedy_fallback,
            translator.stats.n_copy_fallback,
        )
        translations = translator.translate(
            sources,
            batch_size=decode_cfg.batch_size,
            beam=decode_cfg.beam_size,
            alpha=decode_cfg.alpha,
            segment_threshold=decode_cfg.segment_threshold,
        )
        stats_after = (
            translator.stats.n_beam,
            translator.stats.n_greedy_fallback,
            translator.stats.n_copy_fallback,
        )
        pred = dict(zip(ids, translations, strict=True))
        result["timings_seconds"][split] = round(time.monotonic() - t0, 3)

        pred_path = out_dir / f"{split}_predictions.json"
        pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

        gold_rows = [
            {
                "id": r["id"],
                "reference": label_by_id[r["id"]]["reference"],
                "slice": label_by_id[r["id"]].get("slice", split),
            }
            for r in inputs
        ]
        official = compute_official_metrics(pred, gold_rows)
        hyps = [pred[i] for i in ids]
        refs = [label_by_id[i]["reference"] for i in ids]

        entry: dict[str, Any] = {
            "n": len(ids),
            "official": official,
            "official_bleu_ci": bootstrap_ci_official(
                hyps, refs, "bleu", decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
            ),
            "official_chrf_ci": bootstrap_ci_official(
                hyps, refs, "chrf", decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
            ),
            "sacrebleu": sacrebleu_metrics(hyps, refs),
            "fallback_counts": {
                "beam": stats_after[0] - stats_before[0],
                "greedy": stats_after[1] - stats_before[1],
                "copy": stats_after[2] - stats_before[2],
            },
        }
        if split == "dev":
            entry["overall_ci"] = bootstrap_official_overall(
                pred, gold_rows, decode_cfg.n_bootstrap, decode_cfg.bootstrap_seed
            )
        result["sets"][split] = entry
        all_pred.update(pred)
        rows_by_split[split] = [
            {"id": r["id"], "source": r["source"], "reference": label_by_id[r["id"]]["reference"]}
            for r in inputs
        ]

    combined = [
        row
        for split in ("e1", "e2", "e3")
        if split in rows_by_split
        for row in rows_by_split[split]
    ]
    if combined:
        result["length_buckets_e1_e2_e3"] = length_bucket_view(combined, all_pred)

    if decode_cfg.comet:
        comet_triples = [
            {"src": row["source"], "mt": all_pred.get(row["id"], ""), "ref": row["reference"]}
            for split in rows_by_split
            for row in rows_by_split[split]
        ]
        comet_out = out_dir / "comet.json"
        try:
            result["comet"] = run_comet(comet_triples, comet_out)
        except Exception as exc:  # noqa: BLE001 - report the exact error, never fabricate a score
            result["comet_error"] = repr(exc)

    (out_dir / "eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# -------------------------------------------------------------------------------------------
# nmt.train's periodic eval hook (spec §11)
# -------------------------------------------------------------------------------------------


@dataclass
class TrainEvalConfig:
    """Config for the cheap, periodic eval hook `nmt.train.train`'s loop calls every
    `eval.eval_every` steps (spec §11). Fixed, seeded subsets (not full splits) keep this
    affordable during training; `smoke.yaml`-style configs pass small `*_n` values.
    """

    tokenizer_path: Path
    e1_n: int = 500
    e2_n: int = 300
    e3_n: int = 300
    max_len_a: float = 1.5
    max_len_b: int = 10
    no_repeat_ngram_size: int = 3
    n_samples_table: int = 20
    seed: int = 1234
    device: str = "cpu"


def build_train_eval_fn(cfg: TrainEvalConfig) -> Callable[[Any, int, Any], dict[str, Any]]:
    """Build an `nmt.train.EvalFn`: greedy BLEU/chrF (the official §8 functions, via
    `load_official_module`) on fixed seeded subsets of E1[cfg.e1_n]/E2[cfg.e2_n], the *whole*
    official dev set by slice, and E3[cfg.e3_n] (reporting only -- every E3 metric key is
    prefixed `e3_reporting_only_` so it can never be mistaken for a selection signal downstream).
    Also logs a W&B Table of `cfg.n_samples_table` fixed sample translations directly to the
    passed-in run (not through the returned dict, which stays pure float/str/int -- safe for the
    training loop's `json.dumps` into metrics.jsonl).
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(cfg.tokenizer_path))
    module = load_official_module()
    device = torch.device(cfg.device)
    rng = random.Random(cfg.seed)

    def _label_rows(split: str, n: int | None) -> list[dict[str, str]]:
        inputs, labels = load_split(split)
        label_by_id = {r["id"]: r for r in labels}
        subset = inputs if n is None or n >= len(inputs) else rng.sample(inputs, n)
        return [
            {
                "id": r["id"],
                "source": r["source"],
                "reference": label_by_id[r["id"]]["reference"],
                "slice": label_by_id[r["id"]].get("slice", split),
            }
            for r in subset
        ]

    fixed_e1 = _label_rows("e1", cfg.e1_n)
    fixed_e2 = _label_rows("e2", cfg.e2_n)
    fixed_e3 = _label_rows("e3", cfg.e3_n)
    fixed_dev = _label_rows("dev", None)  # always the whole (150-sentence) dev set
    sample_pool = fixed_dev if len(fixed_dev) >= cfg.n_samples_table else fixed_e1
    sample_rows = (
        sample_pool
        if len(sample_pool) <= cfg.n_samples_table
        else rng.sample(sample_pool, cfg.n_samples_table)
    )

    @torch.no_grad()
    def _greedy_translate(model: Any, rows: list[dict[str, str]]) -> dict[str, str]:
        if not rows:
            return {}
        ids_list = [sp.encode(r["source"], out_type=int) for r in rows]
        src, src_mask = _pad_ids(ids_list, model.cfg.pad_id, model.cfg.eos_id, device)
        hyps = greedy_decode(
            model,
            src,
            src_mask,
            model.cfg.bos_id,
            model.cfg.eos_id,
            model.cfg.pad_id,
            max_len_a=cfg.max_len_a,
            max_len_b=cfg.max_len_b,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        )
        return {
            r["id"]: _detok(sp, h.tokens, model.cfg.eos_id) for r, h in zip(rows, hyps, strict=True)
        }

    def eval_fn(model: Any, step: int, wandb_run: Any = None) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        metrics: dict[str, Any] = {}

        for name, rows in (("e1", fixed_e1), ("e2", fixed_e2)):
            pred = _greedy_translate(model, rows)
            scores = module.score_slice(
                [pred[r["id"]] for r in rows], [r["reference"] for r in rows]
            )
            metrics[f"{name}_bleu"], metrics[f"{name}_chrf"], metrics[f"{name}_n"] = (
                scores["bleu"],
                scores["chrf"],
                scores["n"],
            )

        pred_dev = _greedy_translate(model, fixed_dev)
        by_slice: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in fixed_dev:
            by_slice[r["slice"]].append((pred_dev[r["id"]], r["reference"]))
        for slice_name, pairs in by_slice.items():
            scores = module.score_slice([p[0] for p in pairs], [p[1] for p in pairs])
            metrics[f"dev_{slice_name}_bleu"] = scores["bleu"]
            metrics[f"dev_{slice_name}_chrf"] = scores["chrf"]
            metrics[f"dev_{slice_name}_n"] = scores["n"]

        pred_e3 = _greedy_translate(model, fixed_e3)
        scores_e3 = module.score_slice(
            [pred_e3[r["id"]] for r in fixed_e3], [r["reference"] for r in fixed_e3]
        )
        metrics["e3_reporting_only_bleu"] = scores_e3["bleu"]
        metrics["e3_reporting_only_chrf"] = scores_e3["chrf"]
        metrics["e3_reporting_only_n"] = scores_e3["n"]

        if wandb_run is not None and sample_rows:
            try:
                import wandb

                pred_samples = _greedy_translate(model, sample_rows)
                table = wandb.Table(columns=["id", "slice", "source", "reference", "hypothesis"])
                for r in sample_rows:
                    table.add_data(
                        r["id"], r["slice"], r["source"], r["reference"], pred_samples[r["id"]]
                    )
                wandb_run.log({"eval/sample_translations": table}, step=step)
            except Exception as exc:  # noqa: BLE001 - W&B failures must never abort training
                metrics["sample_table_error"] = repr(exc)

        if was_training:
            model.train()
        return metrics

    return eval_fn
