from __future__ import annotations

# Tests for nmt/compare.py: paired A/B comparison over two evaluate runs' prediction files
# (spec §8). Uses monkeypatched tiny splits so no real eval data or model is needed.
import json
from pathlib import Path

import numpy as np
import pytest

from nmt.compare import (
    _load_preds,
    _objective_inputs,
    _objective_point,
    _objective_samples,
    compare_eval_dirs,
    main,
)
from nmt.evaluate import _bleu_sentence_stats, _chrf_sentence_stats
from nmt.selection import (
    SelectionSet,
    limited_selection_set,
    load_selection_set,
    selection_objective,
)

_LABELS = [
    {"id": "d0", "reference": "The cat sat on the mat.", "slice": "seen"},
    {"id": "d1", "reference": "It is raining in Paris today.", "slice": "seen"},
    {"id": "d2", "reference": "We will meet tomorrow morning.", "slice": "long"},
    {"id": "d3", "reference": "The report was published last week.", "slice": "long"},
]
_GOOD = {r["id"]: r["reference"] for r in _LABELS}
_BAD = {r["id"]: "the the the" for r in _LABELS}


@pytest.fixture
def two_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr("nmt.compare.load_split", lambda name: ([], _LABELS))
    a, b = tmp_path / "a", tmp_path / "b"
    for d, preds in ((a, _GOOD), (b, _BAD)):
        d.mkdir()
        (d / "dev_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    return a, b


def test_better_system_has_positive_delta_and_small_p(two_runs: tuple[Path, Path]) -> None:
    a, b = two_runs
    result = compare_eval_dirs(a, b, splits=("dev",), n_bootstrap=200)
    chrf = result["splits"]["dev"]["overall"]["chrf"]
    assert chrf["delta"] > 0
    assert chrf["ci_low"] <= chrf["delta"] <= chrf["ci_high"]
    assert chrf["p_value"] < 0.05
    assert set(result["splits"]["dev"]["by_slice"]) == {"seen", "long"}


def test_identical_systems_have_zero_delta(two_runs: tuple[Path, Path]) -> None:
    a, _ = two_runs
    result = compare_eval_dirs(a, a, splits=("dev",), n_bootstrap=100)
    for metric in ("bleu", "chrf"):
        assert result["splits"]["dev"]["overall"][metric]["delta"] == 0


def test_missing_ids_are_an_error(two_runs: tuple[Path, Path]) -> None:
    a, b = two_runs
    (b / "dev_predictions.json").write_text(json.dumps({"d0": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        compare_eval_dirs(a, b, splits=("dev",), n_bootstrap=10)


def test_cli_writes_report(two_runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = two_runs
    out = tmp_path / "cmp.json"
    assert main(["--a", str(a), "--b", str(b), "--out", str(out), "--splits", "dev"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["splits"]["dev"]["n"] == 4


# ---- E2-synth inclusion --------------------------------------------------------------------

_SYNTH_LABELS = [
    {"id": "s0", "reference": "The cat sat on the mat. It purred.", "slice": "e2synth_400_600"},
    {"id": "s1", "reference": "It is raining in Paris today.", "slice": "e2synth_400_600"},
    {"id": "s2", "reference": "We will meet tomorrow morning.", "slice": "e2synth_600_800"},
    {"id": "s3", "reference": "The report was published last week.", "slice": "e2synth_600_800"},
]


def _patch_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nmt.compare.load_split",
        lambda name: ([], _SYNTH_LABELS if name == "e2synth" else _LABELS),
    )


def test_e2synth_auto_included_when_both_dirs_have_it(
    two_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = two_runs
    _patch_splits(monkeypatch)
    for d, good in ((a, True), (b, False)):
        preds = {r["id"]: (r["reference"] if good else "the the the") for r in _SYNTH_LABELS}
        (d / "e2synth_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    result = compare_eval_dirs(a, b, splits=("dev",), n_bootstrap=100)
    entry = result["splits"]["e2synth"]
    assert entry["synthetic"] is True and entry["label"] == "E2-synth (synthetic)"
    assert set(entry["by_slice"]) == {"e2synth_400_600", "e2synth_600_800"}
    assert entry["by_slice"]["e2synth_600_800"]["chrf"]["delta"] > 0
    assert "synthetic" not in result["splits"]["dev"]
    off = compare_eval_dirs(a, b, splits=("dev",), n_bootstrap=10, include_e2synth=False)
    assert set(off["splits"]) == {"dev"}


def test_e2synth_skipped_when_only_one_dir_has_it(
    two_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = two_runs
    _patch_splits(monkeypatch)
    (a / "e2synth_predictions.json").write_text(json.dumps({"s0": "x"}), encoding="utf-8")
    result = compare_eval_dirs(a, b, splits=("dev",), n_bootstrap=10)
    assert set(result["splits"]) == {"dev"}


# ---- selection-objective paired bootstrap --------------------------------------------------


def _objective_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n1: int = 40, n2: int = 25
) -> tuple[Path, Path, SelectionSet, SelectionSet]:
    """Two eval dirs over a verified PREFIX of the real E1/E2 (so `selection_objective` accepts
    them): A = references with every 4th hypothesis damaged, B = A with every 2nd damaged."""
    e1 = limited_selection_set(load_selection_set("e1"), n1)
    e2 = limited_selection_set(load_selection_set("e2"), n2)

    def fake_load_split(name: str) -> tuple[list[dict], list[dict]]:
        s = {"e1": e1, "e2": e2}[name]
        inputs = [{"id": i, "source": src} for i, src in zip(s.ids, s.sources, strict=True)]
        labels = [
            {"id": i, "reference": r, "slice": s.name}
            for i, r in zip(s.ids, s.references, strict=True)
        ]
        return inputs, labels

    monkeypatch.setattr("nmt.compare.load_split", fake_load_split)
    dirs = []
    for name, step in (("a", 4), ("b", 2)):
        d = tmp_path / name
        d.mkdir()
        for s, split in ((e1, "e1"), (e2, "e2")):
            preds = {
                i: ("the the the" if k % step == 0 else r)
                for k, (i, r) in enumerate(zip(s.ids, s.references, strict=True))
            }
            (d / f"{split}_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
        dirs.append(d)
    return dirs[0], dirs[1], e1, e2


def test_objective_point_equals_selection_objective_difference_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, e1, e2 = _objective_dirs(tmp_path, monkeypatch)
    result = compare_eval_dirs(a, b, splits=(), n_bootstrap=100, objective=True)
    obj = result["selection_objective"]
    hyps_a = {**_load_preds(a, "e1"), **_load_preds(a, "e2")}
    hyps_b = {**_load_preds(b, "e1"), **_load_preds(b, "e2")}
    expected_a = selection_objective(hyps_a, e1, e2)
    expected_b = selection_objective(hyps_b, e1, e2)
    assert obj["a"] == expected_a  # every component bit-for-bit, not just the total
    assert obj["b"] == expected_b
    assert obj["delta"] == expected_a["objective"] - expected_b["objective"]
    assert obj["delta"] > 0  # A damages fewer hypotheses than B
    assert (obj["n_e1"], obj["n_e2"]) == (40, 25)
    assert 0.0 <= obj["p_value"] <= 1.0 and obj["ci_low"] <= obj["ci_high"]


def test_vectorized_objective_matches_official_on_identity_resample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resampling with the identity index (each sentence once) must reproduce the official
    objective: guards the sufficient-statistics formula against the scorer, not just itself."""
    a, _, e1, e2 = _objective_dirs(tmp_path, monkeypatch)
    h1, r1 = _objective_inputs(a, "e1")
    h2, r2 = _objective_inputs(a, "e2")
    got = _objective_samples(
        _bleu_sentence_stats(h1, r1),
        _chrf_sentence_stats(h1, r1),
        _bleu_sentence_stats(h2, r2),
        _chrf_sentence_stats(h2, r2),
        np.arange(len(r1))[None, :],
        np.arange(len(r2))[None, :],
    )
    assert got[0] == pytest.approx(_objective_point(h1, r1, h2, r2)["objective"], abs=1e-9)


def test_objective_bootstrap_identical_systems_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = _objective_dirs(tmp_path, monkeypatch)
    same = compare_eval_dirs(a, a, splits=(), n_bootstrap=50, objective=True)["selection_objective"]
    assert same["delta"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    assert same["p_value"] == 1.0
    r1 = compare_eval_dirs(a, b, splits=(), n_bootstrap=50, objective=True)
    r2 = compare_eval_dirs(a, b, splits=(), n_bootstrap=50, objective=True)
    assert r1["selection_objective"] == r2["selection_objective"]


def test_objective_cli_flag_writes_selection_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = _objective_dirs(tmp_path, monkeypatch)
    out = tmp_path / "cmp.json"
    argv = [
        "--a",
        str(a),
        "--b",
        str(b),
        "--out",
        str(out),
        "--splits",
        "e1",
        "--n-bootstrap",
        "20",
    ]
    assert main([*argv, "--objective"]) == 0
    assert "selection_objective" in json.loads(out.read_text(encoding="utf-8"))
    assert main(argv) == 0
    assert "selection_objective" not in json.loads(out.read_text(encoding="utf-8"))
