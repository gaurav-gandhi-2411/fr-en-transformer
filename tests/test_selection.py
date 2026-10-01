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
import nmt.tune as tune_module
from nmt.selection import (
    SelectionSet,
    limited_selection_set,
    load_selection_set,
    select,
    selection_objective,
)

SELECTION_PY = Path(selection_module.__file__)
TUNE_PY = Path(tune_module.__file__)


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


def test_select_rejects_hand_built_set_named_e1_with_foreign_content() -> None:
    """The dangerous bypass: a SelectionSet declared as "e1" but filled with sentences from a
    reporting-only set (here: the real E3 proxy files). The name check alone would pass it."""
    rows_in = [
        json.loads(line)
        for line in (selection_module.REPO_ROOT / "data/eval/e3/inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows_lab = [
        json.loads(line)
        for line in (selection_module.REPO_ROOT / "data/eval/e3/labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    real_e1, real_e2 = load_selection_set("e1"), load_selection_set("e2")
    smuggled = SelectionSet(
        name="e1",
        sources=tuple(r["source"] for r in rows_in),
        references=tuple(r["reference"] for r in rows_lab),
        ids=tuple("e1_" + r["id"] for r in rows_in),
        inputs_sha256=real_e1.inputs_sha256,
        labels_sha256=real_e1.labels_sha256,
    )
    hyps = dict.fromkeys(smuggled.ids + real_e2.ids, "x")
    with pytest.raises(TypeError, match="canonical"):
        select({"cand": hyps}, smuggled, real_e2)
    with pytest.raises(TypeError, match="canonical"):
        selection_objective(hyps, smuggled, real_e2)
    # A one-sentence edit of the genuine set is refused too.
    edited = dataclasses.replace(real_e1, references=("tampered",) + real_e1.references[1:])
    with pytest.raises(TypeError, match="canonical"):
        select({"cand": hyps}, edited, real_e2)


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


def test_limited_selection_set_returns_unchanged_when_no_limit_or_limit_too_big() -> None:
    e1 = load_selection_set("e1")
    assert limited_selection_set(e1, None) == e1
    assert limited_selection_set(e1, len(e1) + 1000) == e1


def test_limited_selection_set_truncates_to_a_verified_prefix() -> None:
    e1 = load_selection_set("e1")
    limited = limited_selection_set(e1, 5)
    assert len(limited) == 5
    assert limited.name == "e1"
    assert limited.ids == e1.ids[:5]
    assert limited.sources == e1.sources[:5]
    assert limited.references == e1.references[:5]
    # sha256 fields keep pointing at the FULL files -- this is a verified prefix, not a
    # different/smaller dataset.
    assert limited.inputs_sha256 == e1.inputs_sha256
    assert limited.labels_sha256 == e1.labels_sha256


def test_limited_selection_set_passes_select_and_scores_only_the_subset() -> None:
    e1 = load_selection_set("e1")
    e2 = load_selection_set("e2")
    limited_e1 = limited_selection_set(e1, 5)
    limited_e2 = limited_selection_set(e2, 5)
    perfect = {i: r for i, r in zip(limited_e1.ids, limited_e1.references, strict=True)}
    perfect.update({i: r for i, r in zip(limited_e2.ids, limited_e2.references, strict=True)})

    result = select({"perfect": perfect}, limited_e1, limited_e2)
    assert result["best"] == "perfect"
    assert result["scores"]["perfect"]["objective"] > 0


def test_limited_selection_set_rejects_a_hand_truncated_set_with_wrong_content() -> None:
    """A smaller SelectionSet whose rows do NOT match the canonical set's prefix (not built via
    `limited_selection_set`) must still be refused -- the guard checks content, not just size."""
    e1 = load_selection_set("e1")
    bogus = dataclasses.replace(
        e1,
        sources=e1.sources[:5],
        references=("tampered",) + e1.references[1:5],
        ids=e1.ids[:5],
    )
    with pytest.raises(TypeError, match="canonical"):
        select({"cand": dict.fromkeys(bogus.ids, "x")}, bogus, load_selection_set("e2"))


_FORBIDDEN_PATTERNS = [
    re.compile(r"\be3\b", re.IGNORECASE),
    re.compile(r"\bdev\w*", re.IGNORECASE),  # catches "dev", "dev_", "dev.jsonl", etc.
    re.compile(r"opus_books", re.IGNORECASE),
    re.compile(r"labels_dev", re.IGNORECASE),
]


_SCANNED_SOURCE_FILES = (SELECTION_PY, TUNE_PY)


@pytest.mark.parametrize("path", _SCANNED_SOURCE_FILES, ids=lambda p: p.name)
def test_selection_source_never_references_other_eval_sets(path: Path) -> None:
    """Source scan (spec §7, §12, §15): nmt/selection.py and nmt/tune.py (the decoding-tuning CLI,
    which selects only through nmt.selection.load_selection_set/select) must contain no reference
    -- in code, comments, or docstrings -- to any eval set other than e1/e2."""
    source = path.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATTERNS:
        hits = pattern.findall(source)
        assert not hits, f"forbidden reference {pattern.pattern!r} found in {path.name}: {hits}"
