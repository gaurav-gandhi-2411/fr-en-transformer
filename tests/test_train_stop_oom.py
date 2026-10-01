from __future__ import annotations

# Contention/OOM robustness of `nmt.train` (RTX 3070 queue): the cooperative STOP_REQUESTED file,
# the OOM abort, the test-only --debug-raise-oom-at-step hook and the resume accounting. Each
# interrupted path must leave state from which the resumed run reproduces the uninterrupted loss
# trajectory (same pattern as tests/test_train_forced_resume.py).
import json
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from nmt.train import (
    DEBUG_OOM_MARKER,
    EX_TEMPFAIL,
    LEDGER_FILE,
    STOP_FILE_NAME,
    WANDB_ID_FILE,
    load_config,
    train,
)
from nmt.train import main as train_main
from tests.test_train_forced_resume import _losses, _write_config

STEPS = 12


def _cfg(tmp_path: Path, name: str, eval_every: int = 100000):  # noqa: ANN202 - TrainConfig
    cfg_path = tmp_path / f"{name}.yaml"
    _write_config(cfg_path, tmp_path / name, tmp_path / f"{name}_ckpt")
    cfg = load_config(cfg_path)
    cfg.optim.planned_steps = STEPS
    cfg.eval.eval_every = eval_every
    return cfg_path, cfg


def _cli(cfg_path: Path, *extra: str) -> None:
    steps = ["--planned-steps", str(STEPS)]
    train_main(["--config", str(cfg_path), "--synthetic", "--wandb", "disabled", *steps, *extra])


def _scramble_rngs() -> None:
    random.seed(999)  # fresh-process stand-in: scramble every RNG train() doesn't own
    np.random.seed(999)
    torch.manual_seed(999)


def _info(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))


def _assert_same_trajectory(a: Path, b: Path) -> None:
    la, lb = _losses(a), _losses(b)
    assert set(la) == set(lb) == set(range(1, STEPS + 1))
    for step in la:
        assert abs(la[step] - lb[step]) < 1e-4, step


def test_stop_request_checkpoints_exits_75_and_resume_matches_uninterrupted(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ref_path, _ = _cfg(tmp_path, "ref")
    _cli(ref_path)
    capsys.readouterr()

    _, cfg = _cfg(tmp_path, "b", eval_every=7)
    run_dir = Path(cfg.logging.run_dir)

    def request_stop(model: object, step: int, wandb_run: object = None) -> dict[str, float]:
        (run_dir / STOP_FILE_NAME).write_text("foreign GPU pid 4242", encoding="utf-8")
        return {}

    with pytest.raises(SystemExit) as exc:
        train(cfg, synthetic=True, wandb_mode="disabled", eval_fn=request_stop)
    assert exc.value.code == EX_TEMPFAIL
    out1 = capsys.readouterr().out
    saved = re.search(r"CKPT_SAVED step=7 rng_fingerprint=([0-9a-f]{12}) path=(\S+)", out1)
    assert saved, out1  # saved through the normal path although ckpt_minutes=999 / no ckpt_steps
    assert "STOPPED_ON_REQUEST step=7 reason=foreign GPU pid 4242" in out1
    assert not (run_dir / STOP_FILE_NAME).exists()  # consumed
    assert sorted(_losses(run_dir)) == list(range(1, 8))
    info = _info(run_dir)
    assert info["exit_reason"] == "stopped_on_request" and info["final_step"] == 7
    assert info["redo_steps"] == 0  # a stop is saved at the boundary: nothing to redo

    _scramble_rngs()
    train(cfg, synthetic=True, wandb_mode="disabled", resume=True, resume_count=1)
    out2 = capsys.readouterr().out
    resumed = re.search(r"RESUMED step=7 rng_fingerprint=([0-9a-f]{12}) matches_saved=(\w+)", out2)
    assert resumed and resumed.group(2) == "True" and resumed.group(1) == saved.group(1)
    _assert_same_trajectory(tmp_path / "ref", run_dir)


def test_stop_request_is_ignored_by_a_run_that_has_finished(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _, cfg = _cfg(tmp_path, "done")
    run_dir = Path(cfg.logging.run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / STOP_FILE_NAME).write_text("late", encoding="utf-8")
    train(cfg, synthetic=True, wandb_mode="disabled", max_steps=1)  # step 1 == target: finished
    assert "STOPPED_ON_REQUEST" not in capsys.readouterr().out
    assert _info(run_dir)["exit_reason"] == "completed"


def test_oom_aborts_without_checkpoint_and_resume_redoes_from_last_ckpt(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ref_path, _ = _cfg(tmp_path, "ref")
    _cli(ref_path)
    capsys.readouterr()

    b_path, _ = _cfg(tmp_path, "b")
    run_dir, ckpt_dir = tmp_path / "b", tmp_path / "b_ckpt"
    args = ("--ckpt-steps", "5", "--debug-raise-oom-at-step", "8")
    with pytest.raises(SystemExit) as exc:
        _cli(b_path, *args, "--resume", "--resume-count", "0")
    assert exc.value.code == EX_TEMPFAIL
    out1 = capsys.readouterr().out
    assert "OOM_ABORT step=8 last_ckpt_step=5" in out1
    assert [p.name for p in ckpt_dir.glob("step_*.pt")] == ["step_00000005.pt"]  # no mid-step save
    assert (run_dir / DEBUG_OOM_MARKER).is_file()
    assert json.loads((run_dir / LEDGER_FILE).read_text())["redo_steps"] == 3
    assert sorted(_losses(run_dir)) == list(range(1, 8))  # the failed step 8 was never logged
    assert _info(run_dir)["exit_reason"] == "oom_abort"

    _scramble_rngs()
    # Same argv again, as the driver does: the hook must NOT fire a second time.
    _cli(b_path, *args, "--resume", "--resume-count", "1", "--wait-seconds", "12.5")
    out2 = capsys.readouterr().out
    assert "OOM_ABORT" not in out2
    assert re.search(r"RESUMED step=5 rng_fingerprint=[0-9a-f]{12} matches_saved=True", out2)
    _assert_same_trajectory(tmp_path / "ref", run_dir)

    info = _info(run_dir)
    assert info["resumed"] is True and info["resume_count"] == 1
    assert info["wait_seconds_total"] == 12.5
    assert info["redo_steps"] == 3 and info["exit_reason"] == "completed"
    assert info["train_wall_seconds"] > 0 and info["final_step"] == STEPS


def test_oom_before_any_checkpoint_restarts_from_scratch_with_redo_equal_to_step(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    b_path, _ = _cfg(tmp_path, "b")
    with pytest.raises(SystemExit):
        _cli(b_path, "--debug-raise-oom-at-step", "4")
    assert "OOM_ABORT step=4 last_ckpt_step=0" in capsys.readouterr().out
    _cli(b_path, "--debug-raise-oom-at-step", "4", "--resume")  # no ckpt: a fresh start
    info = _info(tmp_path / "b")
    assert info["resumed"] is False and info["redo_steps"] == 4


def test_wandb_run_id_is_reused_across_resume_and_summary_carries_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inits: list[dict] = []
    runs: list[types.SimpleNamespace] = []

    def fake_init(**kw: object) -> types.SimpleNamespace:
        inits.append(kw)
        run = types.SimpleNamespace(
            summary_calls=[],
            config_calls=[],
            finished=0,
            log=lambda *a, **k: None,
            finish=lambda: setattr(run, "finished", run.finished + 1),
        )
        run.summary = types.SimpleNamespace(update=run.summary_calls.append)
        run.config = types.SimpleNamespace(
            update=lambda d, allow_val_change=False: run.config_calls.append(d)
        )
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    _, cfg = _cfg(tmp_path, "w", eval_every=7)
    run_dir = Path(cfg.logging.run_dir)

    def request_stop(model: object, step: int, wandb_run: object = None) -> dict[str, float]:
        (run_dir / STOP_FILE_NAME).write_text("x", encoding="utf-8")
        return {}

    with pytest.raises(SystemExit):
        train(cfg, synthetic=True, wandb_mode="offline", resume=True, eval_fn=request_stop)
    run_id = (run_dir / WANDB_ID_FILE).read_text(encoding="utf-8")
    assert inits[0]["id"] == run_id and inits[0]["resume"] == "allow"
    assert runs[0].finished == 1  # W&B closed on the exit-75 path too

    train(
        cfg,
        synthetic=True,
        wandb_mode="offline",
        resume=True,
        resume_count=1,
        wait_seconds=30.0,
    )
    assert inits[1]["id"] == run_id  # one ablation == one W&B run
    final = runs[1].summary_calls[-1]
    assert final["resume_count"] == 1 and final["wait_seconds_total"] == 30.0
    assert final["resumed"] is True and final["exit_reason"] == "completed"
    assert {"train_wall_seconds", "redo_steps"} <= set(final)
    assert runs[1].config_calls[-1] == {"resume_count": 1, "wait_seconds_total": 30.0}
    assert inits[1]["config"]["resume_count"] == 1

    train(cfg, synthetic=True, wandb_mode="offline", resume=False, max_steps=1)
    assert inits[2]["id"] != run_id  # a fresh (non-resume) start must not hijack the old run
