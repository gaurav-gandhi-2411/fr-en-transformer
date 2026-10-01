from __future__ import annotations

# The *_3070 configs must differ from their originals ONLY in the documented, config-only ways
# (name/group/dirs, precision, and the per-config items below) -- same model, data, seed,
# tokens_per_step and schedule shape -- so a 3070 result is comparable to the original recipe.
import copy
from pathlib import Path

import pytest
import yaml

from nmt.train import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# 3070 name -> (original, extra dotted keys allowed to differ beyond the common set)
# Ablations: micro-batch, step budget and safety cap from the 3070 pilot; eval cadence from the
# resulting step time; checkpoint retention cut to the single final checkpoint (PREREG compares
# final checkpoints only, and C: had 4.2 GB free -- 5 + decay-phase checkpoints at ~0.6 GB each
# per run would not fit); ckpt_minutes 15 -> 5 so an OOM abort or a contention stop (the GPU is
# shared; scripts/ablation_3070.py resumes) redoes at most ~5 minutes of training.
_ABLATION_KEYS = {
    "batch.max_tokens",
    "optim.planned_steps",
    "optim.max_minutes",
    "eval.eval_every",
    "ckpt.keep_last",
    "ckpt.keep_decay_phase",
    "ckpt.ckpt_minutes",
}
PAIRS = {
    "pilot_3070": ("pilot", {"batch.max_tokens", "ckpt.keep_last"}),
    "s1_sin_3070": ("s1_sin", _ABLATION_KEYS),
    "s2_rope_3070": ("s2_rope", _ABLATION_KEYS),
    "s3_rope_concat_3070": ("s3_rope_concat", _ABLATION_KEYS),
    "main_ext_3070": ("main", {"batch.max_tokens", "optim.planned_steps", "optim.max_minutes"}),
}
GROUPS = {
    "pilot_3070": "pilot_3070",
    "s1_sin_3070": "ablation_3070",
    "s2_rope_3070": "ablation_3070",
    "s3_rope_concat_3070": "ablation_3070",
    "main_ext_3070": "main_ext_3070",
}
COMMON = {"name", "group", "precision", "ckpt.dir", "logging.run_dir"}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("new", sorted(PAIRS))
def test_3070_config_differs_from_original_only_in_documented_keys(new: str) -> None:
    orig, extra = PAIRS[new]
    a, b = _flatten(_load(orig)), _flatten(_load(new))
    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert differing <= COMMON | extra, sorted(differing - COMMON - extra)
    assert b["precision"] == "bf16"
    assert b["name"] == new
    assert b["ckpt.dir"] == f"runs/{new}/ckpt" and b["logging.run_dir"] == f"runs/{new}"
    assert b["group"] == GROUPS[new]
    for must_match in ("seed", "batch.tokens_per_step", "model.pos", "optim.cooldown_frac"):
        assert a[must_match] == b[must_match]
    load_config(CONFIGS / f"{new}.yaml")  # parses and validates


def test_ablation_and_pilot_and_main_ext_specifics() -> None:
    for new in ("s1_sin_3070", "s2_rope_3070", "s3_rope_concat_3070"):
        cfg = copy.deepcopy(load_config(CONFIGS / f"{new}.yaml"))
        assert cfg.optim.max_minutes == 60  # safety cap only; planned_steps ends the run
        assert cfg.group == "ablation_3070"
    assert load_config(CONFIGS / "pilot_3070.yaml").optim.max_minutes == 15
    # Time-based only, like pilot.yaml: step checkpoints would pollute the throughput measurement.
    assert load_config(CONFIGS / "pilot_3070.yaml").ckpt.ckpt_steps is None
    main_ext, main = load_config(CONFIGS / "main_ext_3070.yaml"), load_config(CONFIGS / "main.yaml")
    assert main_ext.optim.planned_steps == 2 * main.optim.planned_steps


def test_ablation_configs_agree_on_everything_that_must_be_identical() -> None:
    """PREREG §3 fairness: S1/S2/S3 differ only in the tested factor (pos / concat augmentation).
    Identical seed + data shards + tokens_per_step + planned_steps => identical data order and
    step count; identical warmup/max_tokens/ckpt cadence keep schedule and resume points equal."""
    flat = {n: _flatten(_load(n)) for n in ("s1_sin_3070", "s2_rope_3070", "s3_rope_concat_3070")}
    shared = (
        "seed",
        "optim.planned_steps",
        "batch.tokens_per_step",
        "batch.max_tokens",
        "optim.warmup_steps",
        "ckpt.ckpt_minutes",
        "ckpt.keep_last",
        "precision",
        *sorted(k for k in flat["s1_sin_3070"] if k.startswith("data.")),
    )
    assert any(k.startswith("data.") for k in shared)
    for key in shared:
        assert len({repr(f.get(key)) for f in flat.values()}) == 1, key
    assert flat["s1_sin_3070"]["optim.planned_steps"] == 2889
    assert flat["s1_sin_3070"]["ckpt.ckpt_minutes"] == 5
    assert flat["s1_sin_3070"]["ckpt.keep_last"] == 1
