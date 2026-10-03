from __future__ import annotations

# Extension mode (PREREG 2026-10-02 rule 4): `train(..., init_from=<ckpt>)` starts a NEW run from
# another run's checkpoint with the explicit WSD window (optim.decay_start / decay_end). The CPU
# tests use a tiny model on the synthetic shards, dropout ON, and compare against an uninterrupted
# run of the same schedule; they prove (a) the LR continues at the stable LR and follows
# wsd_lr_scale, (b) data cursor and RNG continue, (c) the optimizer state is restored, (d) branches
# start from the named stable checkpoint without touching it, (e) the parent run is never written.
import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import nmt.train as train_module
from nmt.data.loader import Batch
from nmt.train import (
    BatchSection,
    CkptSection,
    EvalSection,
    LoggingSection,
    ModelSection,
    OptimSection,
    OverfitWatch,
    TrainConfig,
    load_config,
    train,
    wsd_lr_scale,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LR, WARMUP = 1e-3, 2


def _cfg(
    root: Path,
    name: str,
    *,
    planned: int,
    decay: tuple[int, int] | None,
    ckpt_steps: int | None = None,
    milestones: list[int] | None = None,
    keep_last: int = 50,
) -> TrainConfig:
    return TrainConfig(
        name=name,
        group="extend_test",
        seed=7,
        device="cpu",
        model=ModelSection(
            vocab_size=32, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32, dropout=0.3
        ),
        batch=BatchSection(max_tokens=128, tokens_per_step=128, chunk_size=32),
        optim=OptimSection(
            lr=LR,
            warmup_steps=WARMUP,
            planned_steps=planned,
            cooldown_frac=0.2,
            decay_start=decay[0] if decay else None,
            decay_end=decay[1] if decay else None,
        ),
        ckpt=CkptSection(
            dir=str(root / name / "ckpt"),
            ckpt_minutes=999,
            ckpt_steps=ckpt_steps,
            keep_last=keep_last,
            keep_decay_phase=False,
            milestone_steps=milestones or [],
        ),
        logging=LoggingSection(run_dir=str(root / name)),
        eval=EvalSection(eval_every=100000),
    )


def _go(cfg: TrainConfig, **kw: object) -> None:
    train(cfg, wandb_mode="disabled", synthetic=True, **kw)  # type: ignore[arg-type]


def _ckpt(root: Path, name: str, step: int) -> Path:
    return root / name / "ckpt" / f"step_{step:08d}.pt"


def _rows(root: Path, name: str) -> dict[int, dict]:
    path = root / name / "metrics.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    return {r["step"]: r for r in rows if "loss" in r and "eval" not in r}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def record_batches(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every micro-batch's example ids, in consumption order, across all runs of one test."""
    seen: list[list[str]] = []
    original = train_module._iter_micro_batches

    def wrapped(sampler: object) -> Iterator[Batch]:
        for batch in original(sampler):  # type: ignore[arg-type]
            seen.append(list(batch.ids))
            yield batch

    monkeypatch.setattr(train_module, "_iter_micro_batches", wrapped)
    return seen


# ---- (a) schedule -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [(30000, 37500), (40000, 50000)],  # Branch A, Branch B (PREREG rule 4)
)
def test_explicit_window_equals_main_wsd_for_the_prereg_branches(start: int, end: int) -> None:
    frac = (end - start) / end  # 0.2 for both: the branches ARE main-style WSD runs of `end` steps
    for step in range(0, end + 5):
        explicit = wsd_lr_scale(step, 4000, end, 0.2, start, end)
        assert explicit == pytest.approx(wsd_lr_scale(step, 4000, end, frac, start), abs=1e-12)
        assert explicit == pytest.approx(wsd_lr_scale(step, 4000, end, 0.2), abs=1e-12)
    assert wsd_lr_scale(start - 1, 4000, end, 0.2, start, end) == 1.0
    assert wsd_lr_scale(start, 4000, end, 0.2, start, end) == 1.0  # first decay step starts at 1.0
    mid = wsd_lr_scale(start + (end - start) // 2, 4000, end, 0.2, start, end)
    assert mid == pytest.approx(0.5)
    assert wsd_lr_scale(end, 4000, end, 0.2, start, end) == 0.0


def test_stable_run_lr_is_constant_from_the_init_step_to_40000() -> None:
    # ext_stable_l4's schedule: warmup 4000, decay window 40,000 -> 50,000; init step 19,000.
    for step in range(19000, 40000):  # 0-indexed LR steps of optimizer steps 19,001 .. 40,000
        assert wsd_lr_scale(step, 4000, 40000, 0.2, 40000, 50000) == 1.0


def test_extension_configs_carry_the_exact_prereg_numbers() -> None:
    stable, a, b = (
        load_config(CONFIGS / f"ext_{n}_l4.yaml") for n in ("stable", "branch_a", "branch_b")
    )
    assert (stable.name, a.name, b.name) == ("ext_stable_l4", "ext_branch_a_l4", "ext_branch_b_l4")
    assert {c.group for c in (stable, a, b)} == {"extend_l4"}
    assert (stable.optim.planned_steps, stable.ckpt.milestone_steps) == (40000, [30000, 40000])
    assert (a.optim.decay_start, a.optim.decay_end, a.optim.planned_steps) == (30000, 37500, 37500)
    assert (b.optim.decay_start, b.optim.decay_end, b.optim.planned_steps) == (40000, 50000, 50000)
    main = load_config(CONFIGS / "main.yaml")
    for cfg in (stable, a, b):
        assert cfg.seed == main.seed == 1234
        assert cfg.model == main.model  # dropout 0.1, architecture
        assert cfg.batch == main.batch  # tokens/step 25,000, micro-batch, concat
        assert cfg.precision == "bf16"
        assert cfg.eval.eval_every == main.eval.eval_every == 500 and cfg.eval.overfit_watch
        for key in ("lr", "betas", "eps", "weight_decay", "grad_clip", "warmup_steps"):
            assert getattr(cfg.optim, key) == getattr(main.optim, key), key
        assert cfg.optim.max_minutes is None
        assert cfg.ckpt.dir == f"runs/{cfg.name}/ckpt" and cfg.logging.run_dir == f"runs/{cfg.name}"
        assert cfg.ckpt.dir != main.ckpt.dir  # never the main run's directory
    assert yaml.safe_load((CONFIGS / "ext_branch_a_l4.yaml").read_text())["name"] == a.name


def test_decay_window_validation(tmp_path: Path) -> None:
    for body, msg in (
        ("optim: {decay_start: 5}", "together"),
        ("optim: {decay_start: 9, decay_end: 5}", "decay_start < decay_end"),
        ("optim: {warmup_steps: 100, decay_start: 5, decay_end: 50}", "warmup_steps <="),
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(f"name: x\n{body}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            load_config(path)


# ---- (b)(c)(d)(e) end-to-end on CPU ------------------------------------------------------------

N_PARENT, N_STABLE, N_A_FROM, N_END = 6, 12, 8, 14  # parent -> stable -> branch A from step 8


def test_init_from_continues_exactly_and_never_writes_to_its_sources(
    tmp_path: Path, record_batches: list[list[str]], capsys: pytest.CaptureFixture
) -> None:
    sched = (N_STABLE, N_STABLE + 4)  # the stable run's schedule: decay would start at its end
    # Reference: ONE uninterrupted run of the stable schedule, 0 -> N_STABLE.
    ref = _cfg(tmp_path, "ref", planned=N_STABLE, decay=sched, ckpt_steps=1)
    _go(ref)
    ref_ids = list(record_batches)
    record_batches.clear()

    # "main": the parent run, stopped at its stable-phase checkpoint N_PARENT.
    main = _cfg(tmp_path, "main", planned=N_STABLE, decay=sched, ckpt_steps=N_PARENT)
    _go(main, max_steps=N_PARENT)
    parent_ids = list(record_batches)
    n_parent = len(parent_ids)
    assert parent_ids == ref_ids[:n_parent]
    record_batches.clear()
    main_before = _tree_hashes(tmp_path / "main")  # (e) snapshot of the parent run directory

    # Scramble every RNG train() does not own: only the checkpoint's RNG state can explain a pass.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    stable = _cfg(tmp_path, "stable", planned=N_STABLE, decay=sched, milestones=[N_A_FROM])
    _go(stable, init_from=_ckpt(tmp_path, "main", N_PARENT))
    out = capsys.readouterr().out
    src_sha = _sha(_ckpt(tmp_path, "main", N_PARENT))
    assert f"INIT_FROM step={N_PARENT} path=" in out and f"sha256={src_sha}" in out
    assert "matches_saved=True" in out and "RESUMED" not in out

    # (b) the data cursor continued: the batches consumed after init-from are exactly the
    #     uninterrupted run's batches from that step on, and the losses/LRs match step by step.
    assert 0 < n_parent < len(ref_ids)
    assert record_batches == ref_ids[n_parent:]  # identical batch ids, in the same order
    ref_rows, stable_rows = _rows(tmp_path, "ref"), _rows(tmp_path, "stable")
    assert sorted(stable_rows) == list(range(N_PARENT + 1, N_STABLE + 1))  # step counter continues
    for step in stable_rows:
        assert stable_rows[step]["loss"] == pytest.approx(ref_rows[step]["loss"], abs=1e-6), step
    # (a) LR: no re-warmup, no restart: at the stable LR for every step after the init step, and
    #     equal to the schedule function (and to the uninterrupted run) at each of them.
    for step, row in stable_rows.items():
        assert row["lr"] == pytest.approx(
            LR * wsd_lr_scale(step - 1, WARMUP, N_STABLE, 0.2, *sched)
        )
        assert row["lr"] == pytest.approx(ref_rows[step]["lr"])
        assert row["lr"] == pytest.approx(LR)
    # (c) the optimizer state was restored, not re-created: AdamW's step counter continued (a
    #     fresh optimizer would read N_STABLE - N_PARENT) and the whole state equals the reference.
    end_ref = torch.load(_ckpt(tmp_path, "ref", N_STABLE), weights_only=False)
    end_stable = torch.load(_ckpt(tmp_path, "stable", N_STABLE), weights_only=False)
    steps = {int(s["step"]) for s in end_stable["optimizer"]["state"].values()}
    assert steps == {N_STABLE}
    for key, value in end_ref["model"].items():
        assert torch.equal(value, end_stable["model"][key]), key
    for idx, st in end_ref["optimizer"]["state"].items():
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(st[name], end_stable["optimizer"]["state"][idx][name]), (idx, name)
    assert end_ref["sampler"] == end_stable["sampler"]

    # (e) the parent run was never written to (every file byte-identical, none added or removed).
    assert _tree_hashes(tmp_path / "main") == main_before

    # (d) Branch A starts from the NAMED stable checkpoint (step N_A_FROM) and leaves it intact.
    stable_before = _tree_hashes(tmp_path / "stable")
    a_cfg = _cfg(tmp_path, "a", planned=N_END, decay=(N_A_FROM, N_END), milestones=[N_END])
    record_batches.clear()
    _go(a_cfg, init_from=_ckpt(tmp_path, "stable", N_A_FROM))
    a_out = capsys.readouterr().out
    a_sha = _sha(_ckpt(tmp_path, "stable", N_A_FROM))
    assert f"INIT_FROM step={N_A_FROM} " in a_out and f"sha256={a_sha}" in a_out
    info = json.loads((tmp_path / "a" / "run_info.json").read_text())
    assert info["init_from_step"] == N_A_FROM and info["init_from_sha256"] == a_sha
    assert info["resumed"] is False
    a_rows = _rows(tmp_path, "a")
    assert sorted(a_rows) == list(range(N_A_FROM + 1, N_END + 1))
    for step, row in a_rows.items():  # linear from the stable LR down to 0 at decay_end
        want = LR * (1.0 - (step - 1 - N_A_FROM) / (N_END - N_A_FROM))
        assert row["lr"] == pytest.approx(want), step
    assert a_rows[N_A_FROM + 1]["lr"] == pytest.approx(LR)  # decay starts at the stable LR
    assert _tree_hashes(tmp_path / "stable") == stable_before
    assert _tree_hashes(tmp_path / "main") == main_before
    # milestone saved and kept; nothing was written into the stable run's ckpt dir
    assert _ckpt(tmp_path, "a", N_END).is_file()

    # Disconnect/resume inside an extension run: --resume picks the run's OWN checkpoint and
    # ignores init_from; the result is the same as an uninterrupted run.
    ref2 = _cfg(tmp_path, "ref2", planned=N_END, decay=(N_A_FROM, N_END))
    _go(ref2, init_from=_ckpt(tmp_path, "stable", N_A_FROM))
    part = _cfg(tmp_path, "part", planned=N_END, decay=(N_A_FROM, N_END))
    _go(part, init_from=_ckpt(tmp_path, "stable", N_A_FROM), max_steps=N_A_FROM + 3)
    capsys.readouterr()
    _go(part, init_from=_ckpt(tmp_path, "stable", N_A_FROM), resume=True)
    resumed_out = capsys.readouterr().out
    assert "--init-from ignored" in resumed_out and f"RESUMED step={N_A_FROM + 3}" in resumed_out
    rows_ref2, rows_part = _rows(tmp_path, "ref2"), _rows(tmp_path, "part")
    assert set(rows_part) == set(rows_ref2)
    for step in rows_ref2:
        assert rows_part[step]["loss"] == pytest.approx(rows_ref2[step]["loss"], abs=1e-6)
        assert rows_part[step]["lr"] == pytest.approx(rows_ref2[step]["lr"])
    assert _tree_hashes(tmp_path / "stable") == stable_before


def test_init_from_continues_exactly_across_epoch_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The synthetic train split (512 examples, 128-token steps) is a few dozen steps per epoch,
    so 400 steps cross several epoch boundaries; the continuation after --init-from at step 150
    must see the same (sampler.epoch, batch ids) sequence and losses as an uninterrupted run."""
    seen: list[tuple[int, tuple[str, ...]]] = []
    original = train_module._iter_micro_batches

    def wrapped(sampler: object) -> Iterator[Batch]:
        for batch in original(sampler):  # type: ignore[arg-type]
            seen.append((sampler.epoch, tuple(batch.ids)))  # type: ignore[attr-defined]
            yield batch

    monkeypatch.setattr(train_module, "_iter_micro_batches", wrapped)
    n, parent_step = 400, 150
    sched = (n, n + 100)
    _go(_cfg(tmp_path, "ref", planned=n, decay=sched))
    ref = list(seen)
    seen.clear()
    _go(
        _cfg(tmp_path, "par", planned=n, decay=sched, ckpt_steps=parent_step), max_steps=parent_step
    )
    parent = list(seen)
    seen.clear()
    _go(
        _cfg(tmp_path, "ext", planned=n, decay=sched), init_from=_ckpt(tmp_path, "par", parent_step)
    )
    ext = list(seen)

    assert len({e for e, _ in ref}) >= 3  # the reference crosses >= 2 epoch boundaries
    assert len({e for e, _ in ext}) >= 3  # ... and so does the continuation itself
    assert parent == ref[: len(parent)]
    assert ext == ref[len(parent) :]  # identical (epoch, ids) sequence, in order
    ref_rows, ext_rows = _rows(tmp_path, "ref"), _rows(tmp_path, "ext")
    assert sorted(ext_rows) == list(range(parent_step + 1, n + 1))
    for step, row in ext_rows.items():
        assert row["loss"] == pytest.approx(ref_rows[step]["loss"], abs=1e-6), step


def test_init_from_guards(tmp_path: Path) -> None:
    main = _cfg(tmp_path, "main", planned=8, decay=(8, 12), ckpt_steps=4)
    _go(main, max_steps=4)
    src = _ckpt(tmp_path, "main", 4)
    with pytest.raises(FileNotFoundError):
        _go(_cfg(tmp_path, "x", planned=8, decay=(8, 12)), init_from=tmp_path / "step_00000099.pt")
    # a run may not initialise into the directory the checkpoint lives in
    with pytest.raises(ValueError, match="inside this run's own ckpt dir"):
        _go(replace(main, ckpt=replace(main.ckpt, dir=str(src.parent))), init_from=src)
    # file name and content must agree on the step
    wrong = tmp_path / "other" / "step_00000005.pt"
    wrong.parent.mkdir()
    wrong.write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="file name says step 5"):
        _go(_cfg(tmp_path, "y", planned=8, decay=(8, 12)), init_from=wrong)
    # the init step must be below the target
    with pytest.raises(ValueError, match="not below"):
        _go(_cfg(tmp_path, "z", planned=4, decay=(4, 8)), init_from=src)
    # not a resume + a non-empty ckpt dir: refuse instead of mixing two runs
    busy = _cfg(tmp_path, "busy", planned=8, decay=(8, 12), ckpt_steps=2)
    _go(busy, max_steps=2)
    with pytest.raises(RuntimeError, match="empty ckpt dir"):
        _go(busy, init_from=src)
    # none of the failed attempts touched the source
    assert src.is_file()


# ---- milestones and retention ------------------------------------------------------------------


def test_milestone_checkpoints_are_saved_off_cadence_and_never_pruned(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "m", planned=12, decay=(12, 16), ckpt_steps=2, milestones=[5], keep_last=2)
    _go(cfg)
    kept = sorted(int(p.stem.split("_")[1]) for p in (tmp_path / "m" / "ckpt").glob("step_*.pt"))
    assert kept == [5, 10, 12]  # 5 is off the every-2 cadence (saved + kept); last 2 = 10, 12


def test_prune_milestones_do_not_consume_keep_last_slots(tmp_path: Path) -> None:
    for step in range(1000, 9000, 1000):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    train_module.prune_checkpoints(tmp_path, 2, None, False, keep_steps=(3000,))
    left = sorted(int(p.stem.split("_")[1]) for p in tmp_path.glob("step_*.pt"))
    assert left == [3000, 7000, 8000]


# ---- overfitting watch -------------------------------------------------------------------------


def test_overfit_watch_flags_two_consecutive_rises_above_the_running_min() -> None:
    w = OverfitWatch()
    seq = [(500, 3.0), (1000, 2.0), (1500, 2.1), (2000, 2.05), (2500, 2.2), (3000, 2.3)]
    flags = [w.update(s, v) for s, v in seq]
    # 2.1 rises (1), 2.05 still above the min 2.0 (2) -> flagged; 2.2 and 2.3 stay flagged
    assert flags == [False, False, False, True, True, True]
    assert (w.min_loss, w.min_step, w.rises) == (2.0, 1000, 4)
    w2 = OverfitWatch()
    assert [w2.update(s, v) for s, v in [(1, 2.0), (2, 2.5), (3, 1.9), (4, 2.0), (5, 2.1)]] == [
        False,
        False,
        False,
        False,
        True,
    ]  # a new minimum resets the streak


def test_overfit_watch_rebuilds_from_metrics_and_dedupes_redone_steps(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"eval": {"step": 500, "val_loss": 2.0}},
        {"step": 600, "loss": 9.9},
        {"eval": {"step": 1000, "val_loss": 2.4}},
        {"eval": {"step": 1000, "val_loss": 1.5}},  # redone after a crash: the last one wins
        {"eval": {"step": 1500, "val_loss": 1.6}},
        {"eval": {"step": 2000, "val_loss": 1.7}},  # beyond upto_step: ignored
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    w = OverfitWatch.from_metrics(path, 1500)
    assert (w.min_loss, w.min_step, w.rises) == (1.5, 1000, 1)
    assert OverfitWatch.from_metrics(tmp_path / "missing.jsonl", 10) == OverfitWatch()


def test_train_reports_overfit_flag_loudly_and_keeps_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    losses = iter([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(train_module, "_eval_val_loss", lambda *a, **k: next(losses))
    cfg = _cfg(tmp_path, "w", planned=8, decay=None)
    cfg = replace(cfg, eval=EvalSection(eval_every=2, overfit_watch=True))
    _go(cfg)
    out = capsys.readouterr().out
    assert out.count("OVERFIT_WATCH FLAG") == 2  # eval at step 6 (2nd rise) and step 8 (3rd rise)
    assert "OVERFIT_WATCH FLAG step=6 val_loss=3.0000 running_min=1.0000 (step 2)" in out
    ev = [
        json.loads(x)["eval"]
        for x in (tmp_path / "w" / "metrics.jsonl").read_text().split("\n")
        if x.startswith('{"eval"')
    ]
    assert [e["overfit_flag"] for e in ev] == [0, 0, 1, 1]
    assert _rows(tmp_path, "w").keys() == set(range(1, 9))  # never auto-stopped


class _FakeRun:
    """Minimal stand-in for a W&B run: a dict summary, no network."""

    def __init__(self) -> None:
        self.summary: dict = {}
        self.config = self

    def update(self, *a: object, **k: object) -> None:
        self.summary.update(a[0] if a and isinstance(a[0], dict) else {})

    def log(self, *a: object, **k: object) -> None:
        pass

    def finish(self) -> None:
        pass


def test_wandb_summary_separates_flagged_ever_from_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run flagged mid-way that recovers to a new minimum: `overfit_flagged_ever` stays True
    (first/last flagged step recorded) while `overfit_flag_current` is False at the end."""
    run = _FakeRun()
    monkeypatch.setattr(train_module, "init_wandb", lambda *a, **k: run)
    # evals at steps 2,4,6,8,10: min 1.0, rise, rise (flag@6), rise (flag@8), new min (clear@10)
    losses = iter([1.0, 2.0, 3.0, 4.0, 0.5])
    monkeypatch.setattr(train_module, "_eval_val_loss", lambda *a, **k: next(losses))
    cfg = _cfg(tmp_path, "w", planned=10, decay=None)
    cfg = replace(cfg, eval=EvalSection(eval_every=2, overfit_watch=True))
    _go(cfg)
    assert run.summary["overfit_flagged_ever"] is True
    assert run.summary["overfit_flag_current"] is False
    assert run.summary["overfit_flag"] is True  # legacy sticky field, unchanged semantics
    assert (run.summary["overfit_flag_first_step"], run.summary["overfit_flag_last_step"]) == (6, 8)
