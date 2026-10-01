from __future__ import annotations

# Tests for nmt/evaluate.py: the harness wrapper's numbers equal the vendored CLI's output
# exactly (spec §12), sacreBLEU signatures, bootstrap CI/paired-bootstrap statistical behavior
# on synthetic systems, the length-bucket view, and a small end-to-end run_evaluation using a
# tiny model. Spec §8, §12.
import json
import math
import random
import time
from pathlib import Path

import pytest
import torch

from nmt.evaluate import (
    EvalRunConfig,
    _bleu_from_aggregated,
    _bleu_resample,
    _bleu_sentence_stats,
    _chrf_resample,
    _chrf_sentence_stats,
    _official_metric_fn,
    _resample_indices,
    bootstrap_ci_by_group,
    bootstrap_ci_official,
    compute_official_metrics,
    length_bucket_label,
    length_bucket_view,
    load_official_module,
    paired_bootstrap,
    run_evaluation,
    run_official_scorer_cli,
    sacrebleu_metrics,
)
from nmt.hub import export_checkpoint
from nmt.model.transformer import ModelConfig, Transformer
from nmt.translate import Translator

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TOKENIZER = REPO_ROOT / "tokenizer" / "spm.model"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


_GOLD_ROWS = [
    {"id": "d1", "reference": "the cat sat on the mat", "slice": "seen"},
    {"id": "d2", "reference": "hello world, how are you?", "slice": "seen"},
    {
        "id": "d3",
        "reference": "a much longer reference sentence to vary bleu behavior a bit",
        "slice": "long",
    },
    {"id": "d4", "reference": "un chat noir", "slice": "unseen_domain"},
    {"id": "d5", "reference": "it was a dark and stormy night", "slice": "unseen_domain"},
]
_PRED = {
    "d1": "the cat sat on the mat",
    "d2": "hello world how are you",
    "d3": "a much shorter sentence",
    "d4": "a black cat",
    "d5": "it was dark and stormy",
}


def test_compute_official_metrics_matches_cli_exactly(tmp_path: Path) -> None:
    gold_path = tmp_path / "labels.jsonl"
    _write_jsonl(gold_path, _GOLD_ROWS)
    pred_path = tmp_path / "preds.json"
    pred_path.write_text(json.dumps(_PRED), encoding="utf-8")
    out_path = tmp_path / "cli_report.json"

    cli_report = run_official_scorer_cli(gold_path, pred_path, out_path)
    harness_report = compute_official_metrics(_PRED, _GOLD_ROWS)

    assert harness_report == cli_report


def test_sacrebleu_metrics_has_score_and_signature() -> None:
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]
    refs = [r["reference"] for r in _GOLD_ROWS]
    result = sacrebleu_metrics(hyps, refs)
    for key in ("bleu", "chrf", "chrf++"):
        assert isinstance(result[key]["score"], float)
        assert isinstance(result[key]["signature"], str) and result[key]["signature"]


def test_bootstrap_ci_brackets_point_estimate() -> None:
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]
    refs = [r["reference"] for r in _GOLD_ROWS]
    ci = bootstrap_ci_official(hyps, refs, "chrf", n_resamples=200, seed=1234)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n"] == len(_GOLD_ROWS)
    assert ci["n_resamples"] == 200


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]
    refs = [r["reference"] for r in _GOLD_ROWS]
    ci1 = bootstrap_ci_official(hyps, refs, "bleu", n_resamples=100, seed=7)
    ci2 = bootstrap_ci_official(hyps, refs, "bleu", n_resamples=100, seed=7)
    assert ci1 == ci2


def test_paired_bootstrap_detects_a_clearly_better_than_b() -> None:
    """System A = perfect hypotheses (== references), system B = obviously wrong. Delta must be
    positive and the one-sided p-value small (spec §12's "test on synthetic systems")."""
    refs = [r["reference"] for r in _GOLD_ROWS]
    hyps_a = list(refs)  # perfect
    hyps_b = ["completely unrelated garbage text zzz" for _ in refs]

    def chrf_fn(h: list[str], r: list[str]) -> float:
        from nmt.evaluate import load_official_module

        module = load_official_module()
        return (
            sum(module.chrf_sentence(a, b) for a, b in zip(h, r, strict=True)) / len(r)
            if r
            else 0.0
        )

    result = paired_bootstrap(hyps_a, hyps_b, refs, chrf_fn, n_resamples=500, seed=1234)
    assert result["delta"] > 0
    assert result["ci_low"] > 0  # entire CI on the positive side: A is robustly better
    assert result["p_value"] < 0.05


def test_paired_bootstrap_identical_systems_has_zero_delta() -> None:
    refs = [r["reference"] for r in _GOLD_ROWS]
    hyps = [_PRED[r["id"]] for r in _GOLD_ROWS]

    def chrf_fn(h: list[str], r: list[str]) -> float:
        from nmt.evaluate import load_official_module

        module = load_official_module()
        return (
            sum(module.chrf_sentence(a, b) for a, b in zip(h, r, strict=True)) / len(r)
            if r
            else 0.0
        )

    result = paired_bootstrap(hyps, hyps, refs, chrf_fn, n_resamples=100, seed=1234)
    assert result["delta"] == 0.0


def test_length_bucket_label_boundaries() -> None:
    assert length_bucket_label(1) == "<=10"
    assert length_bucket_label(10) == "<=10"
    assert length_bucket_label(11) == "11-20"
    assert length_bucket_label(20) == "11-20"
    assert length_bucket_label(21) == "21-40"
    assert length_bucket_label(40) == "21-40"
    assert length_bucket_label(41) == "41-80"
    assert length_bucket_label(80) == "41-80"
    assert length_bucket_label(81) == ">80"
    assert length_bucket_label(500) == ">80"


def test_length_bucket_view_groups_by_word_count() -> None:
    rows = [
        {"id": "a", "source": "one two three", "reference": "one two three"},
        {"id": "b", "source": " ".join(["word"] * 15), "reference": "word " * 15},
        {"id": "c", "source": "another short one", "reference": "another short one"},
    ]
    pred = {"a": "one two three", "b": "word " * 15, "c": "another short one"}
    view = length_bucket_view(rows, pred)
    assert set(view.keys()) == {"<=10", "11-20"}
    assert view["<=10"]["n"] == 2
    assert view["11-20"]["n"] == 1
    for label in view:
        for metric in ("bleu", "chrf"):
            ci = view[label][f"{metric}_ci"]
            assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


# ---- per-slice bootstrap CIs (spec §8: "per slice and metric") ----


def test_bootstrap_ci_by_group_brackets_point_per_slice() -> None:
    ids = [r["id"] for r in _GOLD_ROWS]
    hyps = [_PRED[i] for i in ids]
    refs = [r["reference"] for r in _GOLD_ROWS]
    slices = [r["slice"] for r in _GOLD_ROWS]
    for metric in ("bleu", "chrf"):
        by_group = bootstrap_ci_by_group(hyps, refs, slices, metric, n_resamples=200, seed=1234)
        assert set(by_group) == {"seen", "long", "unseen_domain"}
        for ci in by_group.values():
            assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_bootstrap_ci_by_group_point_estimates_equal_official_cli_by_slice_exactly(
    tmp_path: Path,
) -> None:
    """Per-slice CI point estimates must equal `official/score.py --out`'s `by_slice` values
    exactly (spec: per-slice CIs must keep the point estimate identical to the official scorer's
    own output -- the CI is attached information, never a different number)."""
    gold_path = tmp_path / "labels.jsonl"
    _write_jsonl(gold_path, _GOLD_ROWS)
    pred_path = tmp_path / "preds.json"
    pred_path.write_text(json.dumps(_PRED), encoding="utf-8")
    out_path = tmp_path / "cli_report.json"
    cli_report = run_official_scorer_cli(gold_path, pred_path, out_path)

    ids = [r["id"] for r in _GOLD_ROWS]
    hyps = [_PRED[i] for i in ids]
    refs = [r["reference"] for r in _GOLD_ROWS]
    slices = [r["slice"] for r in _GOLD_ROWS]

    for metric in ("bleu", "chrf"):
        by_group = bootstrap_ci_by_group(hyps, refs, slices, metric, n_resamples=200, seed=1234)
        for slice_name, ci in by_group.items():
            assert math.isclose(
                ci["point"], cli_report["by_slice"][slice_name][metric], abs_tol=1e-9
            )


# ---- small end-to-end run_evaluation, monkeypatched to tiny in-memory eval splits ----

pytestmark_tokenizer = pytest.mark.skipif(
    not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)"
)


def _export_tiny_translator(tmp_path: Path) -> Translator:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16000, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32)
    model = Transformer(cfg)
    ckpt_path = tmp_path / "step_00000010.pt"
    torch.save({"step": 10, "model": model.state_dict()}, ckpt_path)
    out_dir = export_checkpoint([ckpt_path], tmp_path / "export", cfg, REAL_TOKENIZER, average=True)
    return Translator.from_pretrained(str(out_dir), device="cpu")


@pytestmark_tokenizer
def test_run_evaluation_end_to_end_tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_dev = (
        [
            {"id": "dev_0", "source": "Bonjour le monde.", "slice": "seen", "length": 20},
            {"id": "dev_1", "source": "Ceci est un test.", "slice": "unseen_domain", "length": 20},
        ],
        [
            {"id": "dev_0", "reference": "Hello world.", "slice": "seen"},
            {"id": "dev_1", "reference": "This is a test.", "slice": "unseen_domain"},
        ],
    )
    tiny_e1 = (
        [{"id": "e1_0", "source": "Au revoir.", "slice": "e1", "length": 10}],
        [{"id": "e1_0", "reference": "Goodbye.", "slice": "e1"}],
    )

    def fake_load_split(name: str) -> tuple[list[dict], list[dict]]:
        return {"dev": tiny_dev, "e1": tiny_e1}[name]

    monkeypatch.setattr("nmt.evaluate.load_split", fake_load_split)

    translator = _export_tiny_translator(tmp_path)
    decode_cfg = EvalRunConfig(
        beam_size=2, alpha=0.6, batch_size=4, n_bootstrap=20, bootstrap_seed=1
    )
    out_dir = tmp_path / "reports" / "smoke" / "ckpt0"
    result = run_evaluation(
        translator, "smoke", "ckpt0", decode_cfg, out_dir=out_dir, splits=("dev", "e1")
    )

    assert (out_dir / "eval.json").is_file()
    assert (out_dir / "dev_predictions.json").is_file()
    assert (out_dir / "e1_predictions.json").is_file()
    assert set(result["sets"].keys()) == {"dev", "e1"}
    assert result["sets"]["dev"]["n"] == 2
    assert "overall_ci" in result["sets"]["dev"]
    assert "length_buckets_e1_e2_e3" in result
    on_disk = json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))
    assert on_disk == result


@pytestmark_tokenizer
def test_run_evaluation_every_slice_and_bucket_ci_brackets_its_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §8: per-slice/per-E-set/per-length-bucket bootstrap CIs. On a small synthetic run,
    every slice (dev's seen/long/unseen_domain, plus each E-set) and every length bucket must have
    `ci_low <= point <= ci_high` for both official BLEU and official chrF."""
    tiny_dev = (
        [
            {"id": "dev_0", "source": "Bonjour le monde.", "slice": "seen", "length": 20},
            {"id": "dev_1", "source": "Ceci est un test.", "slice": "unseen_domain", "length": 20},
            {"id": "dev_2", "source": "Quelle belle journee.", "slice": "long", "length": 20},
        ],
        [
            {"id": "dev_0", "reference": "Hello world.", "slice": "seen"},
            {"id": "dev_1", "reference": "This is a test.", "slice": "unseen_domain"},
            {"id": "dev_2", "reference": "What a beautiful day.", "slice": "long"},
        ],
    )
    tiny_e1 = (
        [{"id": "e1_0", "source": "Au revoir.", "slice": "e1", "length": 10}],
        [{"id": "e1_0", "reference": "Goodbye.", "slice": "e1"}],
    )
    tiny_e2 = (
        [{"id": "e2_0", "source": "Bonne nuit.", "slice": "e2", "length": 10}],
        [{"id": "e2_0", "reference": "Good night.", "slice": "e2"}],
    )
    tiny_e3 = (
        [{"id": "e3_0", "source": "Il etait une fois.", "slice": "e3", "length": 18}],
        [{"id": "e3_0", "reference": "Once upon a time.", "slice": "e3"}],
    )

    def fake_load_split(name: str) -> tuple[list[dict], list[dict]]:
        return {"dev": tiny_dev, "e1": tiny_e1, "e2": tiny_e2, "e3": tiny_e3}[name]

    monkeypatch.setattr("nmt.evaluate.load_split", fake_load_split)

    translator = _export_tiny_translator(tmp_path)
    decode_cfg = EvalRunConfig(
        beam_size=2, alpha=0.6, batch_size=4, n_bootstrap=20, bootstrap_seed=1
    )
    out_dir = tmp_path / "reports" / "smoke" / "ckpt0"
    result = run_evaluation(
        translator,
        "smoke",
        "ckpt0",
        decode_cfg,
        out_dir=out_dir,
        splits=("dev", "e1", "e2", "e3"),
    )

    assert set(result["sets"]["dev"]["official_ci_by_slice"]["bleu"]) == {
        "seen",
        "long",
        "unseen_domain",
    }
    for split, entry in result["sets"].items():
        for metric in ("bleu", "chrf"):
            for slice_name, ci in entry["official_ci_by_slice"][metric].items():
                assert ci["ci_low"] <= ci["point"] <= ci["ci_high"], (
                    f"{split}/{metric}/{slice_name}"
                )

    assert result["length_buckets_e1_e2_e3"]
    for bucket, scores in result["length_buckets_e1_e2_e3"].items():
        for metric in ("bleu", "chrf"):
            ci = scores[f"{metric}_ci"]
            assert ci["ci_low"] <= ci["point"] <= ci["ci_high"], f"bucket {bucket}/{metric}"


# ---- vectorized bootstrap: sufficient statistics must equal the official functions exactly ----
# (performance fix, spec §8 -- recomputing BLEU/chrF from strings on every resample is what made
# 1000 resamples on E1 (1940 sentences) take several minutes per split per metric; see PLAN.md's
# smoke-gate deviation note.)

_BLEU_STATS_CASES = {
    "normal text": (
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you",
            "a much shorter sentence entirely different",
            "short ref",
            "une petite phrase simple",
        ],
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you today",
            "a much longer reference sentence to vary bleu behavior a bit more than the others",
            "short ref",
            "une petite phrase simple pour tester",
        ],
    ),
    "empty hypothesis among the set": (
        [
            "",
            "hello world how are you",
            "a much shorter sentence",
            "short ref",
            "une petite phrase",
        ],
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you today",
            "a much longer reference sentence to vary bleu behavior a bit more than the others",
            "short ref",
            "une petite phrase simple pour tester",
        ],
    ),
    "all-empty hyps": (
        ["", "", "", "", ""],
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you today",
            "a much longer reference sentence to vary bleu behavior a bit more than the others",
            "short ref",
            "une petite phrase simple pour tester",
        ],
    ),
    "hyps shorter than refs (BP<1)": (
        ["the quick", "hello world", "a much", "short", "une petite"],
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you today",
            "a much longer reference sentence to vary bleu behavior a bit more than the others",
            "short ref",
            "une petite phrase simple pour tester",
        ],
    ),
    "hyps longer than refs": (
        [
            "the quick brown fox jumps over the lazy dog and then some extra words appended",
            "hello world how are you today and here is more filler text",
            "a much longer reference sentence to vary bleu behavior a bit more than the others "
            "plus even more padding words at the end",
            "short ref with extra padding words tacked on",
            "une petite phrase simple pour tester avec des mots en plus a la fin",
        ],
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you today",
            "a much longer reference sentence to vary bleu behavior a bit more than the others",
            "short ref",
            "une petite phrase simple pour tester",
        ],
    ),
    "zero 4-gram matches": (
        [
            "zzz yyy xxx www vvv uuu",
            "qqq rrr sss ttt uuu vvv",
            "mmm nnn ooo ppp qqq rrr sss",
            "aaa bbb ccc ddd",
            "eee fff ggg hhh iii",
        ],
        [
            "the quick brown fox jumps over the lazy dog",
            "hello world how are you today",
            "a much longer reference sentence to vary bleu behavior a bit more than the others",
            "short ref",
            "une petite phrase simple pour tester",
        ],
    ),
}


@pytest.mark.parametrize("case", list(_BLEU_STATS_CASES), ids=list(_BLEU_STATS_CASES))
def test_bleu_sufficient_stats_identity_equals_official_corpus_bleu(case: str) -> None:
    """Summing the per-sentence sufficient statistics over the full (identity) index set and
    applying `_bleu_from_aggregated` must equal `official.corpus_bleu` exactly (to 1e-9), for the
    normal case plus every edge case the formula special-cases (spec's fix description)."""
    hyps, refs = _BLEU_STATS_CASES[case]
    module = load_official_module()
    stats = _bleu_sentence_stats(hyps, refs)
    got = float(
        _bleu_from_aggregated(
            stats.match.sum(axis=0),
            stats.total.sum(axis=0),
            stats.hyp_len.sum(),
            stats.ref_len.sum(),
        )
    )
    want = module.corpus_bleu(hyps, refs)
    assert math.isclose(got, want, abs_tol=1e-9), f"{case}: got {got} want {want}"


@pytest.mark.parametrize("case", list(_BLEU_STATS_CASES), ids=list(_BLEU_STATS_CASES))
def test_chrf_sufficient_stats_identity_equals_official_mean(case: str) -> None:
    """`_chrf_sentence_stats(...).mean()` over the full set must equal the official corpus chrF
    (`module.score_slice(...)["chrf"]`, a sentence average) exactly (to 1e-9)."""
    hyps, refs = _BLEU_STATS_CASES[case]
    module = load_official_module()
    chrf_vals = _chrf_sentence_stats(hyps, refs)
    got = float(chrf_vals.mean())
    want = module.score_slice(hyps, refs)["chrf"]
    assert math.isclose(got, want, abs_tol=1e-9), f"{case}: got {got} want {want}"


def _synthetic_corpus(n: int, seed: int) -> tuple[list[str], list[str]]:
    vocab = [f"w{i}" for i in range(60)]
    rng = random.Random(seed)
    hyps, refs = [], []
    for _ in range(n):
        ref_len = rng.randint(4, 30)
        ref = " ".join(rng.choice(vocab) for _ in range(ref_len))
        if rng.random() < 0.1:
            hyp = ""  # occasional empty hypothesis, exercising the hyp_len==0 edge case
        else:
            hyp_words = ref.split()
            # perturb: drop/duplicate/shuffle a few tokens so hyp != ref most of the time
            if rng.random() < 0.5 and len(hyp_words) > 2:
                del hyp_words[rng.randrange(len(hyp_words))]
            hyp_words += [rng.choice(vocab) for _ in range(rng.randint(0, 3))]
            rng.shuffle(hyp_words)
            hyp = " ".join(hyp_words)
        hyps.append(hyp)
        refs.append(ref)
    return hyps, refs


def test_vectorized_bleu_resamples_match_official_on_resampled_strings() -> None:
    """For 30 random resamples, the vectorized BLEU resample must equal calling
    `official.corpus_bleu` directly on the resampled string lists (to 1e-9) -- spec's required
    test of the vectorization itself, not just the identity/full-set case."""
    module = load_official_module()
    hyps, refs = _synthetic_corpus(40, seed=11)
    stats = _bleu_sentence_stats(hyps, refs)
    idx = _resample_indices(len(hyps), 30, seed=2024)
    vectorized = _bleu_resample(stats, idx)
    for row in range(idx.shape[0]):
        sampled = idx[row].tolist()
        want = module.corpus_bleu([hyps[i] for i in sampled], [refs[i] for i in sampled])
        assert math.isclose(float(vectorized[row]), want, abs_tol=1e-9), f"resample {row}"


def test_vectorized_chrf_resamples_match_official_on_resampled_strings() -> None:
    """Same as above, for chrF."""
    module = load_official_module()
    hyps, refs = _synthetic_corpus(40, seed=12)
    chrf_vals = _chrf_sentence_stats(hyps, refs)
    idx = _resample_indices(len(hyps), 30, seed=2025)
    vectorized = _chrf_resample(chrf_vals, idx)
    for row in range(idx.shape[0]):
        sampled = idx[row].tolist()
        want = module.score_slice([hyps[i] for i in sampled], [refs[i] for i in sampled])["chrf"]
        assert math.isclose(float(vectorized[row]), want, abs_tol=1e-9), f"resample {row}"


def test_paired_bootstrap_vectorized_fast_path_matches_manual_official_resampling() -> None:
    """`paired_bootstrap` with the official `_VectorizableMetric` (the fast numpy path) must
    produce the same per-resample deltas as manually resampling with the same indices and calling
    `official.corpus_bleu` on the resulting string lists for both systems."""
    module = load_official_module()
    hyps_a, refs = _synthetic_corpus(30, seed=21)
    hyps_b, _ = _synthetic_corpus(30, seed=22)
    n_resamples = 30
    seed = 777
    idx = _resample_indices(len(refs), n_resamples, seed)
    want_deltas = sorted(
        module.corpus_bleu([hyps_a[i] for i in row], [refs[i] for i in row])
        - module.corpus_bleu([hyps_b[i] for i in row], [refs[i] for i in row])
        for row in idx.tolist()
    )
    result = paired_bootstrap(
        hyps_a, hyps_b, refs, _official_metric_fn("bleu"), n_resamples=n_resamples, seed=seed
    )
    lo_idx = int(0.025 * n_resamples)
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    assert math.isclose(result["ci_low"], want_deltas[lo_idx], abs_tol=1e-9)
    assert math.isclose(result["ci_high"], want_deltas[hi_idx], abs_tol=1e-9)


def test_bootstrap_1000_resamples_on_2000_sentences_is_fast() -> None:
    """Spec's performance requirement: 1000 resamples on ~2000 sentences must run in well under a
    few seconds, not the several minutes the old string-recomputing implementation took on E1's
    1940 sentences. Generous 5s bound (this repo's CPU-only CI gate, not a tight perf assertion)."""
    hyps, refs = _synthetic_corpus(2000, seed=99)
    for metric in ("bleu", "chrf"):
        t0 = time.perf_counter()
        bootstrap_ci_official(hyps, refs, metric, n_resamples=1000, seed=1234)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"{metric} bootstrap took {elapsed:.2f}s, expected well under 5s"
