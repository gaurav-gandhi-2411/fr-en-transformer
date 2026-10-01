from __future__ import annotations

# Tests for scripts/length_distributions.py: the summary statistics and group collection on tiny
# synthetic repo layouts (no real data needed).
import json
from pathlib import Path

import pytest

from scripts.length_distributions import collect_groups, describe, render_markdown


def test_describe_min_median_p90_max() -> None:
    d = describe(list(range(1, 11)))  # 1..10
    assert d == {"n": 10, "min": 1, "median": 5.5, "p90": pytest.approx(9.1), "max": 10}
    assert describe([7]) == {"n": 1, "min": 7, "median": 7.0, "p90": 7.0, "max": 7}
    with pytest.raises(ValueError):
        describe([])


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_collect_groups_splits_dev_slices_and_synth_buckets(tmp_path: Path) -> None:
    def rows(prefix: str, lens: list[int], slc: list[str] | None = None) -> list[dict]:
        return [
            {"id": f"{prefix}{i}", "source": "x" * n, "slice": (slc or [prefix] * len(lens))[i]}
            for i, n in enumerate(lens)
        ]

    dev = rows("d", [10, 20, 300], ["seen", "unseen_domain", "long"])
    _write(tmp_path / "data/dev/inputs.jsonl", dev)
    _write(
        tmp_path / "data/dev/labels.jsonl",
        [{"id": r["id"], "reference": "y", "slice": r["slice"]} for r in dev],
    )
    _write(tmp_path / "data/test/inputs.jsonl", rows("t", [5, 6]))
    for name in ("e1", "e2", "e3"):
        _write(tmp_path / f"data/eval/{name}/inputs.jsonl", rows(name, [100, 200, 300]))
    _write(
        tmp_path / "data/eval/e2synth/inputs.jsonl",
        rows("s", [450, 650], ["e2synth_400_600", "e2synth_600_800"]),
    )
    g = collect_groups(tmp_path)
    assert g["dev_long_slice"] == [300] and g["dev_all"] == [10, 20, 300]
    assert g["dev_seen_slice"] == [10] and g["dev_unseen_domain_slice"] == [20]
    assert g["e2synth_400_600"] == [450] and g["e2synth_600_800"] == [650]
    assert g["test"] == [5, 6] and g["e2"] == [100, 200, 300]
    md = render_markdown({k: describe(v) for k, v in g.items()})
    assert "| e2synth_400_600 | 1 | 450 | 450 | 450 | 450 |" in md
