from __future__ import annotations

# The committed safety-net submission must stay valid and byte-identical to the verified HF pull
# (submission/README.md records the sha256). Spec §12.
import hashlib
import json
from pathlib import Path

from nmt.submission import DEFAULT_SAMPLE_PATH, validate_submission

SUBMISSION = Path(__file__).resolve().parents[1] / "submission" / "test_predictions_v1.json"
# sha256 of main's test_predictions.json at HF revision c3d8598 (= validation.json pred_sha256).
EXPECTED_SHA256 = "a7078d9a07dd9daad2ffd84197446fe3687bcd9efd6565375b19ceec4f61f361"


def test_submission_v1_validates_against_sample() -> None:
    assert validate_submission(SUBMISSION) == {"n_ids": 330}


def test_submission_v1_330_ids_no_empty() -> None:
    data = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    sample = json.loads(DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8"))
    assert set(data) == set(sample) and len(data) == 330
    assert all(isinstance(v, str) and v.strip() for v in data.values())


def test_submission_v1_bytes_pinned() -> None:
    assert hashlib.sha256(SUBMISSION.read_bytes()).hexdigest() == EXPECTED_SHA256
