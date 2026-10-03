from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mirror_wandb", Path(__file__).resolve().parent.parent / "scripts" / "mirror_wandb.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mw = importlib.util.module_from_spec(_SPEC)
sys.modules["mirror_wandb"] = mw
_SPEC.loader.exec_module(mw)

SOURCE_CONFIG = {
    "model": {"d_model": 512, "n_heads": 8},
    "optim": {"lr": 0.0007, "max_minutes": 240, "planned_steps": 24645},
    "batch": {"max_tokens": 4096, "tokens_per_step": 25000, "num_workers": 0},
    "logging": {"run_dir": "C:\\Users\\someone\\runs", "wandb_entity": "someone"},
    "ckpt": {"dir": "/content/drive/MyDrive/x"},
    "platform": "Linux-6.6",
    "seed": 1234,
    "gpu_name": "NVIDIA L4",
}


def test_filter_config_keeps_only_whitelist() -> None:
    out = mw.filter_config(SOURCE_CONFIG)
    assert out["model"] == {"d_model": 512, "n_heads": 8}
    assert out["optim"] == {"lr": 0.0007, "planned_steps": 24645}  # max_minutes dropped
    assert out["batch"] == {"max_tokens": 4096, "tokens_per_step": 25000}
    assert out["seed"] == 1234 and out["gpu_name"] == "NVIDIA L4"
    for forbidden in ("logging", "ckpt", "platform"):
        assert forbidden not in out


def test_filter_config_ignores_missing_paths() -> None:
    assert mw.filter_config({"unrelated": 1}) == {}
    assert mw.filter_config({"optim": 3}, ["optim.lr"]) == {}  # parent is not a mapping


def test_scalar_row_drops_non_numeric_and_bookkeeping() -> None:
    row = {
        "_step": 5,
        "_runtime": 1.0,
        "_timestamp": 2.0,
        "loss": 2.5,
        "step": 5,
        "optimizer_stepped": True,
        "eval/sample_translations": {"_type": "table-file"},
        "note": "text",
        "bad": float("nan"),
        "inf": float("inf"),
        "none": None,
    }
    assert mw.scalar_row(row) == {"loss": 2.5, "step": 5}


def test_iter_mirror_rows_keeps_steps_and_skips_empty_and_backwards() -> None:
    rows = [
        {"_step": 0, "loss": 1.0},
        {"_step": 1, "note": "no scalars"},
        {"_step": 2, "loss": 0.9},
        {"_step": 1, "loss": 5.0},  # goes backwards: W&B would reject it
        {"loss": 0.1},  # no _step
        {"_step": 3, "loss": 0.8},
    ]
    assert list(mw.iter_mirror_rows(rows)) == [
        (0, {"loss": 1.0}),
        (2, {"loss": 0.9}),
        (3, {"loss": 0.8}),
    ]


def test_label_text() -> None:
    assert mw.mirror_notes("5d9fc85d") == "mirrored from private run 5d9fc85d"
    assert mw.mirror_tag("5d9fc85d") == "mirrored-from-private-run-5d9fc85d"
    assert mw.mirror_run_id("5d9fc85d") == "m-5d9fc85d"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("made for the challenge sponsor", "sponsor_name"),
        ("a.b@example.com", "email"),
        ("C:\\Users\\x", "windows_user_path"),
        ("host Legion", "host_name"),
        ("/content/drive/MyDrive", "drive_path"),
        ("dev_12345", "dev_test_id"),
    ],
)
def test_scan_text_detects_each_pattern(text: str, key: str) -> None:
    assert mw.scan_text(text)[key] >= 1


def test_scan_text_clean_text_and_literals() -> None:
    clean = '{"loss": 2.5, "url": "https://wandb.ai/gauravgandhi429-gaurav-gandhi/p"}'
    assert sum(mw.scan_text(clean).values()) == 0  # entity name in URLs is acceptable
    counts = mw.scan_text("x une phrase secrete y", ["une phrase secrete", "absent"])
    assert counts["dev_test_literal"] == 1


def test_merge_counts_sums_keywise() -> None:
    assert mw.merge_counts({"a": 1}, {"a": 2, "b": 3}) == {"a": 3, "b": 3}


class _FakeApi:
    """Stands in for wandb.Api: serves canned GraphQL answers, records mutations."""

    def __init__(self, access: str | None, after_create: str = "PRIVATE") -> None:
        self.access = access
        self.after_create = after_create
        self.mutations = 0
        self._service_api = self

    def execute_graphql(self, query: str, variables: dict) -> dict:
        if "mutation" in query:
            self.mutations += 1
            assert variables["input"]["access"] == "PRIVATE"
            self.access = self.after_create
            return {"result": {}}
        if self.access is None:
            return {"project": None}
        return {"project": {"id": "x", "name": "p", "entityName": "e", "access": self.access}}


def test_ensure_private_project_creates_when_missing() -> None:
    api = _FakeApi(None)
    info = mw.ensure_private_project(api, "e", "p")
    assert api.mutations == 1 and info["created_this_run"] is True


def test_ensure_private_project_refuses_non_private() -> None:
    with pytest.raises(RuntimeError, match="not PRIVATE"):
        mw.ensure_private_project(_FakeApi("OPEN"), "e", "p")
    with pytest.raises(RuntimeError, match="not PRIVATE"):
        mw.ensure_private_project(_FakeApi(None, after_create="OPEN"), "e", "p")
