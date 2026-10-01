from __future__ import annotations

# Tests for nmt/submission.py's validate_submission: exactly the 330 ids, no empty strings,
# valid UTF-8 JSON object. Spec §12.
import json
from pathlib import Path

import pytest

from nmt.submission import DEFAULT_SAMPLE_PATH, validate_submission

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_ids() -> list[str]:
    sample = json.loads(DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8"))
    return list(sample.keys())


def test_valid_submission_passes(tmp_path: Path) -> None:
    ids = _sample_ids()
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({i: "hello" for i in ids}), encoding="utf-8")
    result = validate_submission(path)
    assert result == {"n_ids": 330}


def test_missing_id_raises(tmp_path: Path) -> None:
    ids = _sample_ids()
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({i: "hello" for i in ids[:-1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="id set does not match"):
        validate_submission(path)


def test_extra_id_raises(tmp_path: Path) -> None:
    ids = _sample_ids()
    data = {i: "hello" for i in ids}
    data["not_a_real_id"] = "hello"
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="id set does not match"):
        validate_submission(path)


def test_empty_string_value_raises(tmp_path: Path) -> None:
    ids = _sample_ids()
    data = {i: "hello" for i in ids}
    data[ids[0]] = ""
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="empty/whitespace-only or non-string"):
        validate_submission(path)


def test_whitespace_only_value_raises(tmp_path: Path) -> None:
    ids = _sample_ids()
    data = {i: "hello" for i in ids}
    data[ids[0]] = "   "
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="empty/whitespace-only or non-string"):
        validate_submission(path)


def test_non_string_value_raises(tmp_path: Path) -> None:
    ids = _sample_ids()
    data = {i: "hello" for i in ids}
    data[ids[0]] = 42
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="empty/whitespace-only or non-string"):
        validate_submission(path)


def test_not_a_json_object_raises(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_submission(path)
