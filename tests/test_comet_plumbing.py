from __future__ import annotations

# Tests for nmt.evaluate.run_comet's subprocess plumbing (spec §8): correct invocation of the
# isolated envs/comet uv project, and honest failure reporting (never fabricates a score). The
# real COMET-22 model (~2.3GB download) is verified manually/once, not in this CI-safe suite --
# see PLAN.md for that result.
import json
import subprocess
from pathlib import Path

import pytest

from nmt.evaluate import COMET_ENV_DIR, COMET_SCRIPT, run_comet

_TRIPLES = [
    {"src": "Bonjour le monde.", "mt": "Hello world.", "ref": "Hello, world."},
    {"src": "Merci.", "mt": "Thanks.", "ref": "Thank you."},
]


def test_run_comet_invokes_isolated_uv_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["cmd"] = cmd
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "model": "Unbabel/wmt22-comet-da",
                    "n": 2,
                    "scores": [0.8, 0.9],
                    "system_score": 0.85,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_path = tmp_path / "comet.json"
    result = run_comet(_TRIPLES, out_path)

    assert captured["cmd"][:4] == ["uv", "run", "--project", str(COMET_ENV_DIR)]
    assert str(COMET_SCRIPT) in captured["cmd"]
    assert result["system_score"] == 0.85
    assert result["n"] == 2


def test_run_comet_raises_with_exact_error_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="ModuleNotFoundError: no comet"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="ModuleNotFoundError: no comet"):
        run_comet(_TRIPLES, tmp_path / "comet.json")
