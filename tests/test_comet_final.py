from __future__ import annotations

# Tests for scripts/comet_final.py's dedup/layout logic (COMET itself is not run here).
import json
from pathlib import Path

from nmt.evaluate import load_split
from scripts.comet_final import RUNS, SPLITS, VARIANTS, assemble, collect_triples, split_chunks


def _fake_out_root(tmp: Path, diverge_run: str | None = None) -> Path:
    for run in RUNS:
        for variant in VARIANTS:
            d = tmp / run / variant
            d.mkdir(parents=True)
            for split in SPLITS:
                inputs, _ = load_split(split)
                suffix = " X" if run == diverge_run else ""
                pred = {r["id"]: r["source"] + suffix for r in inputs}
                (d / f"{split}_predictions.json").write_text(json.dumps(pred), encoding="utf-8")
    return tmp


def test_collect_triples_scores_identical_triples_once(tmp_path: Path) -> None:
    root = _fake_out_root(tmp_path)
    unique, layout = collect_triples(root)
    total = sum(len(v) for v in layout.values())
    # every (run, variant) holds the same predictions -> one distinct triple per sentence
    assert len(unique) <= total // (len(RUNS) * len(VARIANTS))
    assert len(layout) == len(RUNS) * len(VARIANTS) * len(SPLITS)
    # layout maps back to the right triple: the hypothesis equals the source here
    inputs, _ = load_split("dev")
    first = layout[("main", "seg_off", "dev")][0]
    assert unique[first][0] == inputs[0]["source"] == unique[first][1]


def test_collect_triples_keeps_diverging_predictions_separate(tmp_path: Path) -> None:
    same, _ = collect_triples(_fake_out_root(tmp_path / "a"))
    diff, _ = collect_triples(_fake_out_root(tmp_path / "b", diverge_run="s2_rope_l4"))
    assert len(diff) > len(same)


def test_split_chunks_cover_range_in_order() -> None:
    assert split_chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert split_chunks(0, 4) == []


def test_assemble_per_split_means() -> None:
    layout = {("r", "v", s): [0, 1] for s in SPLITS}
    out = assemble([0.5, 1.5], layout, "r", "v")
    assert out["system_score_by_split"]["dev"] == 1.0
    assert out["per_split_scores"]["e3"] == [0.5, 1.5]
