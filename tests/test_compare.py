from __future__ import annotations

# Tests for nmt/compare.py: paired A/B comparison over two evaluate runs' prediction files
# (spec §8). Uses monkeypatched tiny splits so no real eval data or model is needed.
import json
from pathlib import Path

import pytest

from nmt.compare import compare_eval_dirs, main

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
