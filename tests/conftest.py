from __future__ import annotations

# Pytest fixtures shared across the test suite. Spec §12.
import re
import subprocess
from pathlib import Path

import pytest

from tests.fixtures import TINY_VOCAB_SIZE, build_tiny_shard_dir

__all__ = ["TINY_VOCAB_SIZE"]

# ---- public-repo guard: the challenge's evaluation package is not redistributed -----------------
# The public repository omits `official/score.py`, the dev/test input and label files, the sample
# submission and `submission/test_predictions_v1.json`. Tests that need them must SKIP there, not
# fail. A test is turned into a skip only if it failed with a missing-file shaped exception
# (FileNotFoundError, a CalledProcessError from the scorer subprocess, or the SHA256SUMS
# "missing file" assertion) AND the exception text names one of those package paths. Any other
# failure, including a hash mismatch on a file that IS present, still fails. Trade-off: a genuine
# bug whose message names such a path in a missing-file shape would be masked, so the skip reason
# is explicit and the skip summary is visible with `-rs`.
# `[/\\]+`: a Windows path inside a repr()'d command line (CalledProcessError's message) has its
# backslashes doubled.
PACKAGE_PATH_RE = re.compile(
    r"official[/\\]+score\.py"
    r"|data[/\\]+(?:dev|test)[/\\]+[\w.\-]*\.jsonl"
    r"|sample_submission\.json"
    r"|test_predictions_v1\.json"
)
_SHA_MISSING_MARKER = "missing file listed in SHA256SUMS"


def _exception_text(exc: BaseException) -> str:
    """The text a missing-file check searches: the message of `exc`, except for a
    CalledProcessError, whose message only echoes the command line (it always names the scorer,
    present or not), so only its captured stderr/stdout count."""
    parts = [] if isinstance(exc, subprocess.CalledProcessError) else [str(exc)]
    if isinstance(exc, FileNotFoundError):
        parts.append(str(exc.filename or ""))
    if isinstance(exc, subprocess.CalledProcessError):
        for stream in (exc.stderr, exc.stdout):
            if isinstance(stream, bytes):
                stream = stream.decode("utf-8", errors="replace")
            parts.append(str(stream or ""))
    return "\n".join(parts)


def is_missing_package_failure(exc: BaseException) -> bool:
    """True if `exc` is a missing-file failure naming a file of the evaluation package."""
    is_shape = isinstance(exc, FileNotFoundError | subprocess.CalledProcessError) or (
        isinstance(exc, AssertionError) and _SHA_MISSING_MARKER in str(exc)
    )
    return is_shape and PACKAGE_PATH_RE.search(_exception_text(exc)) is not None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # noqa: ANN201
    """Report a missing-evaluation-package failure as a skip (see the block comment above)."""
    report = yield
    if report.failed and call.excinfo is not None and call.when in ("setup", "call"):
        exc = call.excinfo.value
        if is_missing_package_failure(exc):
            first = _exception_text(exc).strip().splitlines()[0]
            report.outcome = "skipped"
            report.longrepr = (
                str(item.path),
                item.location[1] or 0,
                f"Skipped: evaluation package file absent ({first[:200]})",
            )
    return report


@pytest.fixture
def tiny_shard_dir(tmp_path: Path) -> Path:
    """A tiny synthetic shard tree (train + e1 splits, vocab_size=64) under tmp_path."""
    return build_tiny_shard_dir(tmp_path / "shards")
