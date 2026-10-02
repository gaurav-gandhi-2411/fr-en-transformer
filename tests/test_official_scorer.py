from __future__ import annotations

# Integrity checks on the vendored official scorer (spec §2, §12).
#
# Checks: (a) official/score.py is byte-identical to the version handed out for the
# challenge, via its sha256; (b) official/SHA256SUMS matches every file it lists;
# (c) the scorer CLI runs end to end and prints an OVERALL line for a perfect
# (id -> reference) submission. The harness-wrapper-equality test belongs to P4,
# once nmt/evaluate.py exists, and is intentionally not written here.
import hashlib
import json
from pathlib import Path

from nmt.evaluate import run_official_scorer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_PY = REPO_ROOT / "official" / "score.py"
SHA256SUMS = REPO_ROOT / "official" / "SHA256SUMS"
DEV_LABELS = REPO_ROOT / "data" / "dev" / "labels.jsonl"

# sha256 of the score.py handed out for the take-home challenge (LF line endings, no CR bytes).
EXPECTED_SCORE_PY_SHA256 = "0e023e486a2a0ca5a2d111b4bf43b0419479b23d4cb79b7789827f35783a8393"


def _sha256_of(path: Path) -> str:
    """Return the lowercase hex sha256 digest of the file at `path`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_score_py_sha256_matches_expected() -> None:
    """official/score.py must be byte-identical to the vendored original."""
    assert _sha256_of(SCORE_PY) == EXPECTED_SCORE_PY_SHA256


def test_sha256sums_matches_every_listed_file() -> None:
    """Every hash recorded in official/SHA256SUMS must match the file on disk."""
    lines = [line for line in SHA256SUMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "official/SHA256SUMS is empty"
    for line in lines:
        expected_hash, rel_path = line.split("  ", 1)
        actual_path = REPO_ROOT / rel_path
        assert actual_path.is_file(), f"missing file listed in SHA256SUMS: {rel_path}"
        assert _sha256_of(actual_path) == expected_hash, f"hash mismatch for {rel_path}"


def test_scorer_cli_runs_and_prints_overall(tmp_path: Path) -> None:
    """A perfect (id -> reference) submission should score, exit 0, and print OVERALL."""
    gold = {}
    with DEV_LABELS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                gold[obj["id"]] = obj["reference"]
    assert gold, "no dev labels found to build a sanity submission from"

    pred_path = tmp_path / "dev_predictions.json"
    pred_path.write_text(json.dumps(gold), encoding="utf-8")

    result = run_official_scorer(DEV_LABELS, pred_path)  # check=True: a non-zero exit raises
    assert "OVERALL" in result.stdout
