from __future__ import annotations

# Submission-format validator: exactly the 330 test ids, no empty/whitespace-only or non-string
# values, valid UTF-8 JSON object. Spec §7, §12.
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = REPO_ROOT / "data" / "test" / "sample_submission.json"


def validate_submission(
    path: str | Path, sample_path: str | Path = DEFAULT_SAMPLE_PATH
) -> dict[str, int]:
    """Validate a submission JSON file against `sample_path`'s id set (spec §12).

    Raises `ValueError` naming the problem on any violation: not a JSON object, an id set that
    doesn't exactly match `sample_path`, or any value that is missing/empty/whitespace-only/not a
    string. Reading with `encoding="utf-8"` also surfaces invalid UTF-8 as a `UnicodeDecodeError`
    (not swallowed). Returns `{"n_ids": ...}` on success.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: submission must be a JSON object keyed by id, got {type(data).__name__}"
        )

    sample = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    expected_ids = set(sample.keys())
    actual_ids = set(data.keys())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"{path}: id set does not match {sample_path} exactly "
            f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})"
        )

    bad = [i for i, v in data.items() if not isinstance(v, str) or v.strip() == ""]
    if bad:
        raise ValueError(
            f"{path}: {len(bad)} empty/whitespace-only or non-string value(s), "
            f"e.g. {sorted(bad)[:5]}"
        )

    return {"n_ids": len(data)}
