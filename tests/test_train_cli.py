from __future__ import annotations

# Tests for nmt/train.py's CLI overrides (--run-dir, --ckpt-dir, --data-dir, --planned-steps),
# added for the P5 Colab notebook's `python -m nmt.train --run-dir ... --ckpt-dir ...
# [--planned-steps N]` invocation (spec §14 P5). Each override is verified to actually change the
# resolved `TrainConfig` field it documents, with everything else left at its default behaviour.
from pathlib import Path

import pytest
import yaml

from nmt.train import main as train_main
from tests.fixtures import TINY_VOCAB_SIZE, build_tiny_shard_dir


def _write_cli_config(path: Path, *, planned_steps: int = 1000) -> None:
    """A tiny config written to disk (not constructed as a `TrainConfig` directly) so the CLI's
    own `load_config` + override path is actually exercised, not just `train()`."""
    cfg = {
        "name": "cli_test",
        "group": "smoke",
        "seed": 7,
        "device": "cpu",
        "model": {
            "vocab_size": TINY_VOCAB_SIZE,
            "d_model": 16,
            "n_heads": 2,
            "enc_layers": 1,
            "dec_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
        },
        "data": {"shard_dir": "does-not-exist-shards", "train_split": "train", "eval_split": "e1"},
        "batch": {"max_tokens": 128, "tokens_per_step": 128, "chunk_size": 32},
        "optim": {
            "lr": 1.0,
            "warmup_steps": 0,
            "planned_steps": planned_steps,
            "cooldown_frac": 0.5,
        },
        "ckpt": {"dir": "does-not-exist-ckpt", "ckpt_minutes": 999, "ckpt_steps": 1},
        "logging": {"run_dir": "does-not-exist-run"},
        "eval": {"eval_every": 100000},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _read_rows(run_dir: Path) -> list[dict]:
    import json

    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_data_dir_override_points_training_at_an_arbitrary_shard_dir(tmp_path: Path) -> None:
    """Without --data-dir, cfg.data.shard_dir ("does-not-exist-shards") would make ShardDataset
    raise FileNotFoundError -- a successful run proves the override was applied."""
    shard_dir = build_tiny_shard_dir(tmp_path / "shards")
    config_path = tmp_path / "cfg.yaml"
    _write_cli_config(config_path)
    run_dir = tmp_path / "run"
    ckpt_dir = tmp_path / "ckpt"

    train_main(
        [
            "--config",
            str(config_path),
            "--data-dir",
            str(shard_dir),
            "--run-dir",
            str(run_dir),
            "--ckpt-dir",
            str(ckpt_dir),
            "--max-steps",
            "2",
            "--wandb",
            "disabled",
        ]
    )

    assert (run_dir / "metrics.jsonl").is_file()


def test_run_dir_and_ckpt_dir_overrides_are_independent(tmp_path: Path) -> None:
    """--run-dir and --ckpt-dir point at two different, unrelated directories (not run_dir/ckpt),
    matching the Colab notebook's Drive layout where checkpoints and metrics can live apart."""
    shard_dir = build_tiny_shard_dir(tmp_path / "shards")
    config_path = tmp_path / "cfg.yaml"
    _write_cli_config(config_path)
    run_dir = tmp_path / "some" / "metrics_root"
    ckpt_dir = tmp_path / "elsewhere" / "ckpt_root"

    train_main(
        [
            "--config",
            str(config_path),
            "--data-dir",
            str(shard_dir),
            "--run-dir",
            str(run_dir),
            "--ckpt-dir",
            str(ckpt_dir),
            "--max-steps",
            "2",
            "--wandb",
            "disabled",
        ]
    )

    assert (run_dir / "metrics.jsonl").is_file()
    assert list(ckpt_dir.glob("step_*.pt"))
    assert not (run_dir / "ckpt").exists()  # proves ckpt-dir isn't just run-dir/ckpt by default


def test_data_dir_not_overridden_still_raises_on_missing_shards(tmp_path: Path) -> None:
    """Default behaviour (no --data-dir) is unchanged: a config pointing at a nonexistent shard
    dir still fails loudly, same as before this override existed."""
    config_path = tmp_path / "cfg.yaml"
    _write_cli_config(config_path)

    with pytest.raises(FileNotFoundError):
        train_main(
            [
                "--config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--ckpt-dir",
                str(tmp_path / "ckpt"),
                "--max-steps",
                "2",
                "--wandb",
                "disabled",
            ]
        )


def test_planned_steps_override_changes_wsd_schedule_independent_of_max_steps(
    tmp_path: Path,
) -> None:
    """--planned-steps changes cfg.optim.planned_steps (the WSD schedule's own horizon), while
    --max-steps independently caps how many steps this invocation actually runs -- both runs stop
    at step 5, but only the overridden one has started its decay phase by then.
    """
    shard_dir = build_tiny_shard_dir(tmp_path / "shards")

    baseline_config = tmp_path / "baseline.yaml"
    _write_cli_config(baseline_config, planned_steps=1000)  # decay starts at step 500: no decay yet
    baseline_run = tmp_path / "baseline_run"
    train_main(
        [
            "--config",
            str(baseline_config),
            "--data-dir",
            str(shard_dir),
            "--run-dir",
            str(baseline_run),
            "--ckpt-dir",
            str(tmp_path / "baseline_ckpt"),
            "--max-steps",
            "5",
            "--wandb",
            "disabled",
        ]
    )
    baseline_lr_at_5 = next(r["lr"] for r in _read_rows(baseline_run) if r.get("step") == 5)
    assert baseline_lr_at_5 == pytest.approx(1.0)  # still fully in the stable phase

    overridden_config = tmp_path / "overridden.yaml"
    _write_cli_config(overridden_config, planned_steps=1000)  # YAML says 1000; CLI overrides to 5
    overridden_run = tmp_path / "overridden_run"
    train_main(
        [
            "--config",
            str(overridden_config),
            "--data-dir",
            str(shard_dir),
            "--run-dir",
            str(overridden_run),
            "--ckpt-dir",
            str(tmp_path / "overridden_ckpt"),
            "--max-steps",
            "5",
            "--planned-steps",
            "5",
            "--wandb",
            "disabled",
        ]
    )
    overridden_lr_at_5 = next(r["lr"] for r in _read_rows(overridden_run) if r.get("step") == 5)
    # cooldown_frac=0.5, planned_steps=5 -> decay_len=2, decay_start=3; step 5 (0-indexed 4) is
    # mid-decay: lr_scale = 1 - (4-3)/2 = 0.5.
    assert overridden_lr_at_5 == pytest.approx(0.5)
    assert overridden_lr_at_5 != baseline_lr_at_5


@pytest.mark.parametrize(("cfg_device", "expected"), [("cpu", "cpu"), ("cuda", "cuda")])
def test_eval_hook_decodes_on_the_training_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_device: str, expected: str
) -> None:
    """Regression: the hook used to be hard-coded to CPU while the model sat on CUDA, which
    crashes at the first eval on Colab (input tensors and weights on different devices)."""
    import nmt.evaluate as evaluate_module
    from nmt.train import build_eval_fn, load_config

    captured = {}
    monkeypatch.setattr(
        evaluate_module, "build_train_eval_fn", lambda c: captured.setdefault("cfg", c)
    )
    cfg_path = tmp_path / "c.yaml"
    _write_cli_config(cfg_path)
    cfg = load_config(cfg_path)
    cfg.device = cfg_device
    build_eval_fn(cfg, seed=3)
    assert captured["cfg"].device == expected
    assert captured["cfg"].seed == 3


def test_cli_wires_the_real_eval_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: `python -m nmt.train` (what the Colab notebook runs) used to pass no eval_fn,
    so W&B got val loss but no BLEU/chrF curves. Only --synthetic keeps the no-op hook."""
    import nmt.train as train_module

    sentinel = object()
    seen: list[object] = []
    monkeypatch.setattr(train_module, "build_eval_fn", lambda cfg, seed: sentinel)
    monkeypatch.setattr(train_module, "train", lambda cfg, **kw: seen.append(kw["eval_fn"]))
    cfg_path = tmp_path / "c.yaml"
    _write_cli_config(cfg_path)
    train_module.main(["--config", str(cfg_path)])
    train_module.main(["--config", str(cfg_path), "--synthetic"])
    assert seen == [sentinel, train_module.default_eval_fn]
