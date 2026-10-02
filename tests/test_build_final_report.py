from __future__ import annotations

# Tests for scripts/build_final_report.py's pure helpers: p/CI formatting, bootstrap mean CI,
# per-slice means and the PREREG §3 verdict block. No pulled artifacts needed.
from pathlib import Path
from typing import Any

import pytest

from scripts.build_final_report import (
    bootstrap_mean_ci,
    fmt_ci,
    fmt_p,
    slice_means,
    verdict_block,
)


def test_fmt_p_never_prints_zero() -> None:
    assert fmt_p(0.0) == "< 0.001"
    assert fmt_p(0.0, 10000) == "< 0.0001"
    assert fmt_p(0.5) == "0.500"


def test_fmt_ci_signed_and_unsigned() -> None:
    assert fmt_ci(1.234, 0.5, 2.0) == "1.23 [0.50, 2.00]"
    assert fmt_ci(1.234, -0.5, 2.0, signed=True) == "+1.23 [-0.50, +2.00]"


def test_bootstrap_mean_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [float(v) for v in range(100)]
    a, b = bootstrap_mean_ci(values), bootstrap_mean_ci(values)
    assert a == b
    assert a["point"] == pytest.approx(49.5)
    assert a["ci_low"] < a["point"] < a["ci_high"]
    assert a["n"] == 100 and a["n_resamples"] == 1000


def test_bootstrap_mean_ci_of_constant_is_degenerate() -> None:
    r = bootstrap_mean_ci([3.0] * 10)
    assert r["ci_low"] == r["ci_high"] == r["point"] == 3.0


def test_slice_means_groups_by_label() -> None:
    out = slice_means([1.0, 1.0, 5.0, 5.0], ["a", "a", "b", "b"])
    assert out["a"]["point"] == 1.0 and out["b"]["point"] == 5.0


def _c(delta: float, lo: float, hi: float, p: float) -> dict[str, float]:
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "p_value": p}


def _comp(e2: dict, syn: dict, e1: dict) -> dict[str, Any]:
    both = lambda c: {"chrf": c, "bleu": c}  # noqa: E731 - tiny test helper
    return {
        "delta": "A - B",
        "splits": {
            "e1": {"overall": both(e1)},
            "e2": {"overall": both(e2)},
            "dev": {
                "overall": both(e2),
                "by_slice": {"seen": both(e2), "long": both(e2)},
            },
            "e2synth": {"overall": both(syn), "by_slice": {"e2synth_400_600": both(syn)}},
        },
        "selection_objective": _c(1.0, 0.5, 1.5, 0.0),
    }


def test_h1_supported_needs_both_e2_and_e2synth() -> None:
    good = _comp(_c(0.9, 0.6, 1.1, 0.0), _c(3.0, 2.4, 3.8, 0.0), _c(0, -1, 1, 0.5))
    assert "**SUPPORTED**" in verdict_block("H1", good, None, "PRIMARY")
    bad = _comp(_c(0.9, 0.6, 1.1, 0.0), _c(3.0, -0.1, 3.8, 0.06), _c(0, -1, 1, 0.5))
    assert "**NOT SUPPORTED**" in verdict_block("H1", bad, None, "PRIMARY")


def test_h2_requires_e1_non_inferiority_bound() -> None:
    sig = (_c(0.9, 0.6, 1.1, 0.0), _c(1.0, 0.5, 1.5, 0.0))
    ok = verdict_block("H2", _comp(*sig, _c(0.1, -0.2, 0.4, 0.3)), None, "PRIMARY")
    assert "**SUPPORTED**" in ok
    # CI lower bound -0.6 breaches the -0.5 non-inferiority margin even with E2/E2-synth significant
    breach = verdict_block("H2", _comp(*sig, _c(-0.2, -0.6, 0.2, 0.7)), None, "PRIMARY")
    assert "**NOT SUPPORTED**" in breach


def test_verdict_p_zero_is_printed_as_bound() -> None:
    good = _comp(_c(0.9, 0.6, 1.1, 0.0), _c(3.0, 2.4, 3.8, 0.0), _c(0, -1, 1, 0.5))
    assert "p < 0.001" in verdict_block("H1", good, None, "PRIMARY")


def test_normalize_and_reindex_converts_crlf_and_refreshes_hashes(tmp_path: Path) -> None:
    import hashlib
    import json

    from scripts.build_final_report import RUNS, normalize_and_reindex

    run_dir = tmp_path / RUNS[0]
    (run_dir / "source").mkdir(parents=True)
    derived = run_dir / "eval.json"
    derived.write_bytes(b'{\r\n "a": 1\r\n}')
    downloaded = run_dir / "source" / "x.json"
    downloaded.write_bytes(b'{\r\n "kept": 1\r\n}')  # verbatim HF bytes: must stay untouched
    entry = {"path": "eval.json", "sha256": "stale", "bytes": 0}
    (run_dir / "index.json").write_text(json.dumps({"artifacts": [entry]}), encoding="utf-8")
    assert normalize_and_reindex(tmp_path) == 1
    assert derived.read_bytes() == b'{\n "a": 1\n}'
    assert downloaded.read_bytes() == b'{\r\n "kept": 1\r\n}'
    art = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))["artifacts"][0]
    assert art["sha256"] == hashlib.sha256(derived.read_bytes()).hexdigest()
    assert art["bytes"] == len(derived.read_bytes())


def test_word_copies_flags_only_long_unshared_source_words() -> None:
    from scripts.build_final_report import word_copies

    # "Meaulnes" is kept by the reference -> not flagged; "maison" is copied and absent -> flagged;
    # "de" is shorter than 4 -> ignored
    out = word_copies("Meaulnes de maison", "Meaulnes de maison", "Meaulnes of house")
    assert out == ["maison"]
    assert word_copies("the house", "la maison", "the house") == []


def test_pick_rule_examples_is_deterministic_and_documented() -> None:
    from scripts.build_final_report import pick_rule_examples

    def row(i: str, ref: str, hyp: str) -> dict[str, str]:
        return {"id": i, "source": "s", "reference": ref, "hyp": hyp, "slice": "x"}

    rows = [
        row("a", "one two three four five", "one two three four five"),
        row("b", "one two three four five six", "one"),  # largest length deviation
        row("c", "one two three four five", "x y z x y z x y z"),  # repetition, ref has none
        row("d", "a b c a b c d e", "q w e q w e q w e q w e"),  # ref repeats -> not a B candidate
    ]
    picks = pick_rule_examples(rows)
    assert [p["id"] for p in picks] == ["b", "c"]
    assert picks == pick_rule_examples(list(reversed(rows)))
    assert picks[1]["id"] != picks[0]["id"]
