from __future__ import annotations

# The *_l4 ablation fallback configs (used only if S1 has not started on the 3070 when the Colab
# main run finishes; PLAN.md 2026-10-02) must equal their *_3070 twins except for the
# hardware-dependent keys, and equal each other except for the tested factor (pos / concat).
import math
from pathlib import Path

import pytest
import yaml

from nmt.train import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STEMS = ("s1_sin", "s2_rope", "s3_rope_concat")
HARDWARE_KEYS = {
    "name",
    "group",
    "ckpt.dir",
    "logging.run_dir",
    "batch.max_tokens",
    "optim.planned_steps",
}
# floor(40 min * 60 s * 0.85 margin / 0.4966 s/step) from reports/pilot_l4/pilot_summary.json
L4_PLANNED_STEPS = math.floor(40 * 60 * 0.85 / 0.4966)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _flat(name: str) -> dict:
    return _flatten(yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8")))


@pytest.mark.parametrize("stem", STEMS)
def test_l4_config_differs_from_3070_twin_only_in_hardware_keys(stem: str) -> None:
    a, b = _flat(f"{stem}_3070"), _flat(f"{stem}_l4")
    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert differing == HARDWARE_KEYS, sorted(differing ^ HARDWARE_KEYS)
    assert b["name"] == f"{stem}_l4" and b["group"] == "ablation_l4"
    assert b["ckpt.dir"] == f"runs/{stem}_l4/ckpt" and b["logging.run_dir"] == f"runs/{stem}_l4"
    assert b["batch.max_tokens"] == 4096  # the L4 pilot's measured micro-batch
    assert b["optim.planned_steps"] == L4_PLANNED_STEPS == 4107
    assert b["precision"] == "bf16"
    load_config(CONFIGS / f"{stem}_l4.yaml")  # parses and validates


def test_l4_configs_agree_with_each_other_except_pos_and_concat() -> None:
    flat = {s: _flat(f"{s}_l4") for s in STEMS}
    allowed = {"name", "ckpt.dir", "logging.run_dir", "model.pos", "batch.concat_prob"}
    allowed |= {"batch.concat_max_len"}  # only s3 sets it (3070 twin test pins the origin)
    keys = set().union(*(f.keys() for f in flat.values()))
    differing = {k for k in keys if len({repr(f.get(k)) for f in flat.values()}) > 1}
    assert differing <= allowed, sorted(differing - allowed)
    assert {"model.pos", "batch.concat_prob"} <= differing
    assert flat["s1_sin"]["model.pos"] == "sinusoidal" and flat["s2_rope"]["model.pos"] == "rope"
    assert flat["s2_rope"]["batch.concat_prob"] == 0.0
    assert flat["s3_rope_concat"]["batch.concat_prob"] > 0.0
