from __future__ import annotations

# Tests for nmt/selection.py: SelectionSet's only constructor path, its hard refusal of anything
# but "e1"/"e2", tampered-id rejection, and a source scan proving the module never references any
# other eval set. Spec §8, §12, §15.
import dataclasses
import json
import re
from pathlib import Path

import pytest

import nmt.selection as selection_module
from nmt.selection import SelectionSet, load_selection_set, select, selection_objective

SELECTION_PY = Path(selection_module.__file__)


def test_load_selection_set_e1_and_e2_succeed() -> None:
    e1 = load_selection_set("e1")
    e2 = load_selection_set("e2")
    assert e1.name == "e1"
    assert e2.name == "e2"
    assert len(e1) > 0 and len(e2) > 0
    assert all(i.startswith("e1_") for i in e1.ids)
    assert all(i.startswith("e2_") for i in e2.ids)
    assert len(e1.inputs_sha256) == 64  # hex sha256


@pytest.mark.parametrize("bad_name", ["e3", "dev", "E1", "train", "/etc/passwd", ""])
def test_load_selection_set_refuses_anything_else(bad_name: str) -> None:
    with pytest.raises(ValueError, match="only accepts"):
        load_selection_set(bad_name)  # type: ignore[arg-type]


def test_selection_set_has_no_public_path_constructor() -> None:
    """There is no way to build a SelectionSet from an arbitrary directory: the dataclass has no
    alternate constructor, and `load_selection_set` takes only a hard-coded `name`, never a path.
    """
    import inspect

    sig = inspect.signature(load_selection_set)
    assert list(sig.parameters) == ["name"]
    field_names = {f.name for f in dataclasses.fields(SelectionSet)}
    assert "path" not in field_names and "dir" not in field_names


def test_select_raises_typeerror_on_a_fabricated_e3_selection_set() -> None:
    """A SelectionSet manually constructed with name="e3" (bypassing load_selection_set
    entirely) must still be rejected by `select`/`selection_objective` -- the guard checks the
    object's declared name, not how it was built."""
    real_e2 = load_selection_set("e2")
    fake_e3 = SelectionSet(
        name="e3",  # type: ignore[arg-type]
        sources=("x",),
        references=("y",),
        ids=("e3_fake",),
        inputs_sha256="0" * 64,
        labels_sha256="0" * 64,
    )
    with pytest.raises(TypeError, match="SelectionSet"):
        select({"cand": {"e3_fake": "y"}}, fake_e3, real_e2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SelectionSet"):
        selection_objective({"e3_fake": "y"}, fake_e3, real_e2)  # type: ignore[arg-type]


def test_select_raises_typeerror_on_non_selection_set_arguments() -> None:
    with pytest.raises(TypeError, match="SelectionSet"):
        select({"cand": {}}, {"not": "a selection set"}, load_selection_set("e2"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SelectionSet"):
        select({"cand": {}}, load_selection_set("e1"), "e2")  # type: ignore[arg-type]


def test_load_selection_set_rejects_tampered_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_dir = tmp_path / "e1"
    fake_dir.mkdir()
    (fake_dir / "inputs.jsonl").write_text(
        json.dumps({"id": "NOT_e1_prefixed", "source": "x", "slice": "e1", "length": 1}) + "\n",
        encoding="utf-8",
    )
    (fake_dir / "labels.jsonl").write_text(
        json.dumps({"id": "NOT_e1_prefixed", "reference": "y", "slice": "e1"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(selection_module._ALLOWED_SETS, "e1", fake_dir)
    with pytest.raises(ValueError, match="not prefixed"):
        load_selection_set("e1")


def test_select_picks_the_higher_scoring_candidate() -> None:
    e1 = load_selection_set("e1")
    e2 = load_selection_set("e2")
    perfect = {i: r for i, r in zip(e1.ids, e1.references, strict=True)}
    perfect.update({i: r for i, r in zip(e2.ids, e2.references, strict=True)})
    garbage = dict.fromkeys(perfect, "zzz completely unrelated garbage")

    result = select({"perfect": perfect, "garbage": garbage}, e1, e2)
    assert result["best"] == "perfect"
    assert result["scores"]["perfect"]["objective"] > result["scores"]["garbage"]["objective"]


_FORBIDDEN_PATTERNS = [
    re.compile(r"\be3\b", re.IGNORECASE),
    re.compile(r"\bdev\w*", re.IGNORECASE),  # catches "dev", "dev_", "dev.jsonl", etc.
    re.compile(r"opus_books", re.IGNORECASE),
    re.compile(r"labels_dev", re.IGNORECASE),
]


def test_selection_source_never_references_other_eval_sets() -> None:
    """Source scan (spec §12): nmt/selection.py must contain no reference -- in code, comments,
    or docstrings -- to any eval set other than e1/e2."""
    source = SELECTION_PY.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATTERNS:
        hits = pattern.findall(source)
        assert not hits, f"forbidden reference {pattern.pattern!r} found in selection.py: {hits}"
