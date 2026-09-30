from __future__ import annotations

# Submission-format checks on the provided sample_submission.json (spec §12).
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUBMISSION = REPO_ROOT / "data" / "test" / "sample_submission.json"
TEST_INPUTS = REPO_ROOT / "data" / "test" / "inputs.jsonl"

EXPECTED_TEST_COUNT = 330


def _test_input_ids() -> set[str]:
    """Return the set of ids present in data/test/inputs.jsonl."""
    ids: set[str] = set()
    with TEST_INPUTS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def test_sample_submission_is_json_object() -> None:
    """sample_submission.json must be a JSON object, not a list or scalar."""
    data = json.loads(SAMPLE_SUBMISSION.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_sample_submission_has_exactly_330_ids() -> None:
    """The sample submission must key exactly the 330 test ids."""
    data = json.loads(SAMPLE_SUBMISSION.read_text(encoding="utf-8"))
    assert len(data) == EXPECTED_TEST_COUNT


def test_sample_submission_ids_equal_test_input_ids() -> None:
    """The submission's id set must equal the id set of data/test/inputs.jsonl exactly."""
    data = json.loads(SAMPLE_SUBMISSION.read_text(encoding="utf-8"))
    assert set(data.keys()) == _test_input_ids()
