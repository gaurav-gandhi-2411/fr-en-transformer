from __future__ import annotations

# Unit tests for the public-repo skip guard in tests/conftest.py (a missing evaluation-package
# file must skip a test; every other failure must still fail).
import subprocess

import pytest

from tests.conftest import is_missing_package_failure


def test_file_not_found_naming_a_package_path_is_a_skip() -> None:
    exc = FileNotFoundError(2, "No such file or directory", "data/dev/inputs.jsonl")
    assert is_missing_package_failure(exc)


def test_scorer_subprocess_failure_is_searched_in_stderr() -> None:
    exc = subprocess.CalledProcessError(
        2, ["python", "official/score.py"], stderr=b"FileNotFoundError: official\\score.py"
    )
    assert is_missing_package_failure(exc)


def test_sha256sums_missing_file_assertion_is_a_skip() -> None:
    exc = AssertionError("missing file listed in SHA256SUMS: data/test/sample_submission.json")
    assert is_missing_package_failure(exc)


def test_unrelated_missing_file_still_fails() -> None:
    assert not is_missing_package_failure(FileNotFoundError("configs/nope.yaml"))


def test_hash_mismatch_on_a_present_package_file_still_fails() -> None:
    assert not is_missing_package_failure(AssertionError("hash mismatch for official/score.py"))


def test_other_exception_types_naming_a_package_path_still_fail() -> None:
    assert not is_missing_package_failure(ValueError("bad row in data/dev/labels.jsonl"))


def test_scorer_failure_not_naming_a_package_path_still_fails() -> None:
    exc = subprocess.CalledProcessError(1, ["scorer"], stderr="KeyError: 'id'")
    assert not is_missing_package_failure(exc)


def test_python_cant_open_file_stderr_has_doubled_backslashes_and_matches() -> None:
    # python reports "can't open file %r", so a Windows path arrives with doubled backslashes
    stderr = "python: can't open file 'C:\\\\repo\\\\official\\\\score.py': [Errno 2] No such file"
    exc = subprocess.CalledProcessError(
        2, ["python", "C:\\repo\\official\\score.py"], stderr=stderr
    )
    assert is_missing_package_failure(exc)


def test_command_line_alone_does_not_count_for_a_scorer_failure() -> None:
    # the scorer ran (it is present) and rejected its input: that is a real failure, not a skip
    exc = subprocess.CalledProcessError(
        1, ["python", "official/score.py", "--gold", "data/dev/labels.jsonl"], stderr="bad input"
    )
    assert not is_missing_package_failure(exc)


@pytest.mark.parametrize("sep", ["/", "\\"])
def test_both_path_separators_match(sep: str) -> None:
    assert is_missing_package_failure(FileNotFoundError(f"official{sep}score.py"))
