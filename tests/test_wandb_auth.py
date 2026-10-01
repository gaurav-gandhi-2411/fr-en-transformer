from __future__ import annotations

# Regression: on Windows, W&B credentials live in ~/_netrc. The online-mode check looked only at
# ~/.netrc, so `--wandb online` silently fell back to offline on the RTX 3070 laptop.
from pathlib import Path

import pytest

from nmt.train import _resolve_wandb_mode, _wandb_authenticated


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("NETRC", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.mark.parametrize("name", [".netrc", "_netrc"])
def test_netrc_or_windows_netrc_counts_as_authenticated(fake_home: Path, name: str) -> None:
    (fake_home / name).write_text("machine api.wandb.ai\n  login user\n  password x\n")
    assert _wandb_authenticated()
    assert _resolve_wandb_mode("online") == "online"


def test_no_credentials_falls_back_to_offline(fake_home: Path) -> None:
    (fake_home / "_netrc").write_text("machine github.com\n  login u\n  password x\n")
    assert not _wandb_authenticated()
    assert _resolve_wandb_mode("online") == "offline"
