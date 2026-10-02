from __future__ import annotations

# CONFIG = "ablations_l4": the notebook runs s1_sin_l4 -> s2_rope_l4 -> s3_rope_concat_l4
# sequentially in one Colab session. All logic lives INSIDE the notebook cells (a Colab session
# clones the v0.2.2-colab tag, which has no new colab/ files), so these tests exec the real cells
# by id, like tests/test_colab_l4.py. `run_step` is replaced by a recorder that simulates what
# nmt.train writes (run_info.json + the final checkpoint), so nothing is ever trained.
import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import nbformat
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "colab" / "train.ipynb"
PARAMS, URL, HELPERS = "c5459372", "f4b8d2a6", "a3c7e1b9"
TRAIN, DRIVE, SUMMARY, QUICK_EVAL = "4e5fb4a0", "d5f9868b", "87d33708", "cfe05e3f"
ORDER = ("s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
PLANNED = 4107
ENTITY, PROJECT = "ent", "proj"


def _src(cell_id: str) -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    return next(c.source for c in nb.cells if c.id == cell_id)


def _exec(cell_id: str, namespace: dict[str, Any], edits: dict[str, str] | None = None) -> None:
    source = _src(cell_id)
    for old, new in (edits or {}).items():
        assert old in source, old
        source = source.replace(old, new, 1)
    exec(compile(source, f"<notebook cell {cell_id}>", "exec"), namespace)  # noqa: S102


def _params(**edits: str) -> dict[str, Any]:
    """Exec the Parameters cell with `NAME = value` lines rewritten (`CONFIG` defaults to
    ablations_l4); returns its namespace.
    """
    values = {"CONFIG": '"ablations_l4"', **edits}
    namespace: dict[str, Any] = {}
    replacements = {}
    for name, literal in values.items():
        line = next(ln for ln in _src(PARAMS).splitlines() if ln.startswith(f"{name} = "))
        replacements[line.split("  #")[0]] = f"{name} = {literal}"
    _exec(PARAMS, namespace, replacements)
    return namespace


class FakeRunStep:
    """Records every run_step call and simulates what nmt.train leaves behind."""

    def __init__(self, outcomes: dict[str, str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outcomes = outcomes or {}

    def __call__(self, step: str, argv: list[str], **kwargs: Any) -> str:
        cfg = Path(argv[argv.index("--config") + 1]).stem
        self.calls.append({"step": step, "argv": list(argv), "cfg": cfg, **kwargs})
        outcome = self.outcomes.get(cfg, "completed")
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        ckpt_dir = Path(argv[argv.index("--ckpt-dir") + 1])
        if outcome == "crash":
            raise RuntimeError(f"{step} failed (exit 1)")
        if outcome == "completed":
            _write_state(run_dir, ckpt_dir, "completed", PLANNED, [PLANNED])
        else:  # stopped_early: exits 0 with the run unfinished
            _write_state(run_dir, ckpt_dir, "stopped_early", 2000, [2000])
        return ""


def _write_state(
    run_dir: Path, ckpt_dir: Path, reason: str | None, final_step: int | None, ckpts: list[int]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if reason is not None:
        info = {"exit_reason": reason, "final_step": final_step, "train_wall_seconds": 12.5}
        (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    for step in ckpts:
        (ckpt_dir / f"step_{step:08d}.pt").write_bytes(b"x")


def _helpers(run_step: Any = None) -> dict[str, Any]:
    namespace: dict[str, Any] = {"Path": Path, "re": re, "run_step": run_step or FakeRunStep()}
    _exec(URL, namespace)
    _exec(HELPERS, namespace)
    return namespace


def _dirs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    run_dirs = {c: tmp_path / "runs" / f"notebook_{c}" for c in ORDER}
    return run_dirs, {c: d / "ckpt" for c, d in run_dirs.items()}


def _sequence(ns: dict[str, Any], tmp_path: Path, *, dry_run: bool = False) -> dict[str, str]:
    run_dirs, ckpt_dirs = _dirs(tmp_path)
    return ns["run_ablation_sequence"](
        ORDER,
        repo_dir=REPO_ROOT,
        run_dirs=run_dirs,
        ckpt_dirs=ckpt_dirs,
        data_dir=tmp_path / "shards",
        wandb_mode="offline",
        preflight={"preflight_gpu": "NVIDIA L4"},
        dry_run=dry_run,
    )


# --- parameters ---------------------------------------------------------------------------------


def test_fixed_order_and_config_files_exist() -> None:
    ns = _params()
    assert ns["ABLATION_CONFIGS"] == ORDER
    assert ns["ABLATION"] is True and "ablations_l4" in ns["ALLOWED_CONFIGS"]
    for name in ORDER:
        assert (REPO_ROOT / "configs" / f"{name}.yaml").is_file()


def test_defaults_stay_the_main_run_and_are_not_ablations() -> None:
    ns = _params(CONFIG='"main"')
    assert ns["PLANNED_STEPS"] == 24645 and ns["ABLATION"] is False and ns["DRY_RUN"] is False


def test_planned_steps_is_ignored_loudly_in_ablations_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = _params()  # PLANNED_STEPS keeps its default 24645
    out = capsys.readouterr().out
    assert ns["PLANNED_STEPS"] is None
    assert "ABLATIONS MODE: PLANNED_STEPS=24645 is IGNORED" in out and "4107" in out


@pytest.mark.parametrize("flag", ["COOLDOWN_NOW", "RESUME_TEST"])
def test_cooldown_now_and_resume_test_raise_in_ablations_mode(flag: str) -> None:
    with pytest.raises(ValueError, match=flag):
        _params(**{flag: "True"})


def test_dry_run_is_refused_outside_ablations_mode() -> None:
    with pytest.raises(ValueError, match="DRY_RUN"):
        _params(CONFIG='"main"', DRY_RUN="True")
    assert _params(DRY_RUN="True")["DRY_RUN"] is True


def test_unknown_config_error_names_ablations_l4() -> None:
    with pytest.raises(ValueError, match="ablations_l4"):
        _params(CONFIG='"s1_sin_l4"')


# --- config contract the notebook relies on -----------------------------------------------------


def _flat(name: str) -> dict[str, Any]:
    def walk(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, dict):
                out.update(walk(value, f"{prefix}{key}."))
            else:
                out[f"{prefix}{key}"] = value
        return out

    text = (REPO_ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
    return walk(yaml.safe_load(text))


def test_regex_scalars_match_the_yaml_parser() -> None:
    read = _helpers()["read_config_scalars"]
    for name in ORDER:
        flat = _flat(name)
        scalars = read(REPO_ROOT / "configs" / f"{name}.yaml")
        assert scalars == {
            "name": flat["name"],
            "group": flat["group"],
            "planned_steps": flat["optim.planned_steps"],
        }


def test_l4_configs_share_group_steps_seed_data_and_model_except_pos() -> None:
    flats = {name: _flat(name) for name in ORDER}
    assert {f["group"] for f in flats.values()} == {"ablation_l4"}
    assert {f["optim.planned_steps"] for f in flats.values()} == {PLANNED}
    assert {f["seed"] for f in flats.values()} == {1234}
    data_keys = {k for f in flats.values() for k in f if k.startswith("data.")}
    for key in data_keys:
        assert len({repr(f.get(key)) for f in flats.values()}) == 1, key
    model_keys = {k for f in flats.values() for k in f if k.startswith("model.")} - {"model.pos"}
    for key in model_keys:
        assert len({repr(f.get(key)) for f in flats.values()}) == 1, key
    assert [flats[n]["model.pos"] for n in ORDER] == ["sinusoidal", "rope", "rope"]


# --- status helper: truth table -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "final_step", "ckpts", "expected"),
    [
        (None, None, [], "fresh"),  # nothing on disk
        ("stopped_early", 0, [], "fresh"),  # an invocation that never checkpointed
        (None, None, [500], "resume"),  # checkpoints but no completion record (crash/disconnect)
        ("stopped_early", 2000, [2000], "resume"),
        ("stopped_on_request", 1500, [1500], "resume"),
        ("oom_abort", 1200, [1000], "resume"),
        (None, None, [PLANNED], "resume"),  # final ckpt written but the run never recorded it
        ("completed", PLANNED, [PLANNED], "finished"),
        ("completed", PLANNED + 5, [PLANNED], "finished"),  # final_step >= planned
        ("completed", PLANNED, [3000], "resume"),  # completed, final checkpoint MISSING
        ("completed", PLANNED, [], "resume"),  # completed, no checkpoint at all
        ("completed", 3000, [PLANNED], "resume"),  # completed record below planned_steps
    ],
)
def test_status_truth_table(
    tmp_path: Path, reason: str | None, final_step: int | None, ckpts: list[int], expected: str
) -> None:
    ns = _helpers()
    run_dir, ckpt_dir = tmp_path / "run", tmp_path / "run" / "ckpt"
    if reason is not None or ckpts:
        _write_state(run_dir, ckpt_dir, reason, final_step, ckpts)
    assert ns["ablation_config_status"](run_dir, ckpt_dir, PLANNED) == expected


def test_status_of_a_corrupt_run_info_is_never_finished(tmp_path: Path) -> None:
    ns = _helpers()
    run_dir, ckpt_dir = tmp_path / "run", tmp_path / "run" / "ckpt"
    _write_state(run_dir, ckpt_dir, None, None, [PLANNED])
    (run_dir / "run_info.json").write_text("{not json", encoding="utf-8")
    assert ns["ablation_config_status"](run_dir, ckpt_dir, PLANNED) == "resume"
    (run_dir / "run_info.json").write_text("[1, 2]", encoding="utf-8")
    assert ns["ablation_config_status"](run_dir, ckpt_dir, PLANNED) == "resume"


def test_completed_without_final_checkpoint_is_flagged(tmp_path: Path) -> None:
    ns = _helpers()
    run_dir, ckpt_dir = tmp_path / "run", tmp_path / "run" / "ckpt"
    _write_state(run_dir, ckpt_dir, "completed", PLANNED, [3000])
    message = ns["ablation_inconsistency"](run_dir, ckpt_dir, PLANNED)
    assert message is not None and "step_00004107.pt" in message and "COMPLETED" in message
    _write_state(run_dir, ckpt_dir, "completed", PLANNED, [PLANNED])
    assert ns["ablation_inconsistency"](run_dir, ckpt_dir, PLANNED) is None


# --- the sequence -------------------------------------------------------------------------------


def _expected_argv(tmp_path: Path, cfg: str) -> list[str]:
    run_dir = tmp_path / "runs" / f"notebook_{cfg}"
    return [
        sys.executable,
        str(REPO_ROOT / "colab" / "run_train.py"),
        str(run_dir / "preflight_config.json"),
        "--config",
        f"configs/{cfg}.yaml",
        "--resume",
        "--wandb",
        "offline",
        "--run-dir",
        str(run_dir),
        "--ckpt-dir",
        str(run_dir / "ckpt"),
        "--data-dir",
        str(tmp_path / "shards"),
    ]


def _mixed_start(tmp_path: Path) -> None:
    """s1 finished, s2 partial (stopped_early at 1000), s3 fresh."""
    run_dirs, ckpt_dirs = _dirs(tmp_path)
    _write_state(run_dirs[ORDER[0]], ckpt_dirs[ORDER[0]], "completed", PLANNED, [PLANNED])
    _write_state(run_dirs[ORDER[1]], ckpt_dirs[ORDER[1]], "stopped_early", 1000, [1000])


def test_sequence_skips_finished_resumes_partial_and_starts_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeRunStep()
    ns = _helpers(fake)
    _mixed_start(tmp_path)
    decisions = _sequence(ns, tmp_path)
    assert decisions == {ORDER[0]: "finished", ORDER[1]: "resume", ORDER[2]: "fresh"}
    assert [c["cfg"] for c in fake.calls] == [ORDER[1], ORDER[2]]  # s1 skipped, order kept
    for call in fake.calls:
        assert call["argv"] == _expected_argv(tmp_path, call["cfg"])
        assert "--planned-steps" not in call["argv"] and "--cooldown-now" not in call["argv"]
        assert call["stream"] is True and call["cwd"] == REPO_ROOT
        run_dir = tmp_path / "runs" / f"notebook_{call['cfg']}"
        assert call["log_path"].parent == run_dir
        assert re.fullmatch(r"train_log_\d{8}T\d{12}Z\.txt", call["log_path"].name)
        preflight = json.loads((run_dir / "preflight_config.json").read_text(encoding="utf-8"))
        assert preflight == {"preflight_gpu": "NVIDIA L4"}
    out = capsys.readouterr().out
    assert "ABLATION 1/3: s1_sin_l4" in out and "decision: finished -> SKIP" in out
    assert "decision: resume" in out and "decision: fresh" in out


def test_a_failed_config_stops_the_sequence_before_the_next_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeRunStep({ORDER[1]: "crash"})
    ns = _helpers(fake)
    _mixed_start(tmp_path)  # s1 finished, s2 partial, s3 fresh
    with pytest.raises(RuntimeError, match=r"s2_rope_l4.*failed \(exit 1\)"):
        _sequence(ns, tmp_path)
    assert [c["cfg"] for c in fake.calls] == [ORDER[1]]  # s3 never started
    assert "ABLATION SEQUENCE STOPPED at s2_rope_l4" in capsys.readouterr().out
    # a later "Run all" (healthy runtime) resumes exactly there: s1 skipped, s2 then s3 run
    healthy = FakeRunStep()
    _sequence(_helpers(healthy), tmp_path)
    assert [c["cfg"] for c in healthy.calls] == [ORDER[1], ORDER[2]]


def test_a_clean_exit_that_is_not_finished_also_stops_the_sequence(tmp_path: Path) -> None:
    fake = FakeRunStep({ORDER[0]: "stopped_early"})  # e.g. the max_minutes safety cap
    with pytest.raises(RuntimeError, match=r"STOPPED at s1_sin_l4.*not finished"):
        _sequence(_helpers(fake), tmp_path)
    assert [c["cfg"] for c in fake.calls] == [ORDER[0]]


def test_completed_record_without_final_checkpoint_is_refused_not_skipped(tmp_path: Path) -> None:
    fake = FakeRunStep()
    run_dirs, ckpt_dirs = _dirs(tmp_path)
    _write_state(run_dirs[ORDER[0]], ckpt_dirs[ORDER[0]], "completed", PLANNED, [3000])
    with pytest.raises(RuntimeError, match="COMPLETED.*missing"):
        _sequence(_helpers(fake), tmp_path)
    assert fake.calls == []


def test_dry_run_launches_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeRunStep()
    ns = _helpers(fake)
    _mixed_start(tmp_path)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    decisions = _sequence(ns, tmp_path, dry_run=True)
    assert fake.calls == []
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert decisions == {ORDER[0]: "finished", ORDER[1]: "resume", ORDER[2]: "fresh"}
    out = capsys.readouterr().out
    assert "DRY_RUN: would skip." in out
    for cfg in ORDER[1:]:
        assert " ".join(_expected_argv(tmp_path, cfg)) in out


# --- evidence + summary -------------------------------------------------------------------------


def _finish_all(ns: dict[str, Any], tmp_path: Path) -> None:
    _sequence(ns, tmp_path)
    for cfg in ORDER:
        (tmp_path / "runs" / f"notebook_{cfg}" / "wandb_run_id.txt").write_text(
            f"id_{cfg}\n", encoding="utf-8"
        )


def test_evidence_lists_every_url_and_final_checkpoint(tmp_path: Path) -> None:
    ns = _helpers()
    _finish_all(ns, tmp_path)
    run_dirs, ckpt_dirs = _dirs(tmp_path)
    rows = ns["collect_ablation_evidence"](
        ORDER,
        repo_dir=REPO_ROOT,
        run_dirs=run_dirs,
        ckpt_dirs=ckpt_dirs,
        entity=ENTITY,
        project=PROJECT,
        online=True,
    )
    text = "\n".join(ns["format_ablation_summary"](rows))
    for cfg in ORDER:
        assert f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/id_{cfg}" in text  # THIS run's URL
        assert str(ckpt_dirs[cfg] / f"step_{PLANNED:08d}.pt") in text
    assert "ALL 3 ABLATIONS FINISHED" in text and "NOT finished" not in text
    assert f"{PLANNED:>8}" in text and "12.5" in text  # planned_steps and train_wall_seconds


def test_evidence_flags_unfinished_configs(tmp_path: Path) -> None:
    ns = _helpers(FakeRunStep({ORDER[1]: "crash"}))
    with pytest.raises(RuntimeError):
        _sequence(ns, tmp_path)
    run_dirs, ckpt_dirs = _dirs(tmp_path)
    rows = ns["collect_ablation_evidence"](
        ORDER,
        repo_dir=REPO_ROOT,
        run_dirs=run_dirs,
        ckpt_dirs=ckpt_dirs,
        entity=ENTITY,
        project=PROJECT,
        online=False,
    )
    text = "\n".join(ns["format_ablation_summary"](rows))
    assert (
        "NOT ALL FINISHED (1/3)" in text and "still pending: s2_rope_l4, s3_rope_concat_l4" in text
    )
    assert "MISSING (expected" in text and "n/a (offline mode)" in text


def _summary_namespace(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    ns = _helpers()
    run_dirs, ckpt_dirs = _dirs(tmp_path)
    ns.update(
        GIT_REF="v0.2.2-colab",
        git_sha="abc123",
        git_describe="v0.2.2-colab",
        DATA_REVISION="rev42",
        CONFIG="ablations_l4",
        ABLATION=True,
        EVAL=False,
        ABLATION_CONFIGS=ORDER,
        DRY_RUN=False,
        PILOT_RESUME_TEST=False,
        preflight={"preflight_gpu": "NVIDIA L4", "preflight_precision": "bf16"},
        run_root=tmp_path / "runs",
        ckpt_dir=None,
        wandb_mode="online",
        WANDB_ENTITY=ENTITY,
        WANDB_PROJECT=PROJECT,
        repo_dir=REPO_ROOT,
        ablation_run_dirs=run_dirs,
        ablation_ckpt_dirs=ckpt_dirs,
        stopped_after_first_ckpt=False,
        wandb_run_url=None,
        resume_test_passed=None,
    )
    ns.update(overrides)
    return ns


def test_summary_cell_in_ablations_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ns = _summary_namespace(tmp_path)
    _finish_all(ns, tmp_path)
    capsys.readouterr()
    _exec(SUMMARY, ns)
    out = capsys.readouterr().out
    for needle in (
        "abc123",
        "v0.2.2-colab",
        "rev42",
        "NVIDIA L4",
        "bf16",
        "W&B group: ablation_l4",
    ):
        assert needle in out, needle
    for cfg in ORDER:
        assert f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/id_{cfg}" in out
        assert f"step_{PLANNED:08d}.pt" in out
    assert "ALL 3 ABLATIONS FINISHED" in out


def test_summary_cell_single_config_path_is_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt_dir = tmp_path / "ckpt"
    _write_state(tmp_path, ckpt_dir, None, None, [200, 1000])
    ns = _summary_namespace(
        tmp_path,
        CONFIG="main",
        ABLATION=False,
        ckpt_dir=ckpt_dir,
        run_root=tmp_path,
        wandb_run_url="https://wandb.ai/e/p/runs/zz",
    )
    _exec(SUMMARY, ns)
    out = capsys.readouterr().out
    assert f"latest checkpoint: {ckpt_dir / 'step_00001000.pt'}" in out
    assert "W&B run URL: https://wandb.ai/e/p/runs/zz" in out and "W&B group" not in out


# --- the other cells ----------------------------------------------------------------------------


def test_quick_eval_cell_is_a_noop_with_a_reason_in_ablations_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*_a: Any, **_k: Any) -> str:
        raise AssertionError("quick eval must not launch in ablations mode")

    _exec(QUICK_EVAL, {"ABLATION": True, "RUN_EVAL": True, "run_step": boom})
    assert "quick eval skipped" in capsys.readouterr().out


def _drive_namespace(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    ns: dict[str, Any] = {
        "Path": Path,
        "MODE": "local",
        "repo_dir": tmp_path,
        "ABLATION": True,
        "EVAL": False,
        "DRY_RUN": False,
        "CONFIG": "ablations_l4",
        "ABLATION_CONFIGS": ORDER,
    }
    ns.update(overrides)
    return ns


def test_drive_cell_local_ablations_makes_one_dir_pair_per_config(tmp_path: Path) -> None:
    ns = _drive_namespace(tmp_path)
    _exec(DRIVE, ns)
    for cfg in ORDER:
        assert ns["ablation_run_dirs"][cfg] == tmp_path / "runs" / f"notebook_{cfg}"
        assert ns["ablation_ckpt_dirs"][cfg] == ns["ablation_run_dirs"][cfg] / "ckpt"
        assert ns["ablation_ckpt_dirs"][cfg].is_dir()


def test_drive_cell_dry_run_creates_nothing(tmp_path: Path) -> None:
    _exec(DRIVE, _drive_namespace(tmp_path, DRY_RUN=True))
    assert not (tmp_path / "runs").exists()


def test_drive_cell_colab_ablations_uses_runs_slash_config_on_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounted: list[str] = []
    fake_drive = types.SimpleNamespace(mount=mounted.append)
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = fake_drive  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns = _drive_namespace(tmp_path, MODE="colab", DRY_RUN=True)  # dry: no mkdir on this host
    _exec(DRIVE, ns)
    base = Path("/content/drive/MyDrive/fr-en-transformer/runs")
    assert mounted == ["/content/drive"]
    assert ns["ablation_run_dirs"] == {c: base / c for c in ORDER}
    assert ns["ablation_ckpt_dirs"] == {c: base / c / "ckpt" for c in ORDER}


def test_drive_cell_single_config_dirs_are_unchanged(tmp_path: Path) -> None:
    ns = _drive_namespace(tmp_path, ABLATION=False, CONFIG="main")
    _exec(DRIVE, ns)
    assert ns["run_root"] == tmp_path / "runs" / "notebook_main"
    assert ns["ckpt_dir"] == ns["run_root"] / "ckpt" and ns["ckpt_dir"].is_dir()


def test_new_cells_import_only_stdlib_and_never_yaml_or_nmt() -> None:
    # Kernel import rule (tests/test_colab_helpers.py covers every cell after the install cell);
    # stated explicitly for the ablation cells, whose config parsing must be regex-only.
    for cell_id in (HELPERS, TRAIN, SUMMARY, DRIVE, PARAMS):
        modules = set()
        for node in ast.walk(ast.parse(_src(cell_id))):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                modules.add((node.module or "").split(".")[0])
        assert modules <= set(sys.stdlib_module_names) | {"google"}, (cell_id, modules)


# --- execute_notebook + CI ------------------------------------------------------------------------


def _load_execute_notebook() -> Any:
    spec = importlib.util.spec_from_file_location(
        "colab_execute_notebook_abl", REPO_ROOT / "colab" / "execute_notebook.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NoopClient:
    executed = 0

    def __init__(self, *_a: Any, **_k: Any) -> None:
        pass

    def execute(self) -> None:
        type(self).executed += 1


def test_execute_notebook_allows_ablations_only_with_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_execute_notebook()
    monkeypatch.setattr(mod, "NotebookClient", _NoopClient)
    abl = ["--set", 'CONFIG="ablations_l4"']
    assert mod.main([str(NOTEBOOK), *abl]) == 2  # a real ablations run on CPU: refused
    assert "without DRY_RUN=True" in capsys.readouterr().err
    assert mod.main([str(NOTEBOOK), *abl, "--set", "DRY_RUN=False"]) == 2
    assert mod.main([str(NOTEBOOK), *abl, "--allow-non-smoke"]) == 2  # no override either
    assert _NoopClient.executed == 0
    assert mod.main([str(NOTEBOOK), *abl, "--set", "DRY_RUN=True"]) == 0
    assert _NoopClient.executed == 1
    # smoke and the refusal of every other config are unchanged
    assert mod.main([str(NOTEBOOK), "--set", 'CONFIG="smoke"']) == 0
    assert mod.main([str(NOTEBOOK), "--set", 'CONFIG="pilot"']) == 2
    assert mod.main([str(NOTEBOOK), "--set", 'CONFIG="pilot"', "--allow-non-smoke"]) == 0


def test_execute_notebook_effective_param_reads_the_override() -> None:
    mod = _load_execute_notebook()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert mod.effective_param(nb, "DRY_RUN") == "False"
    mod.apply_overrides(nb, {"DRY_RUN": "True"})
    assert mod.effective_param(nb, "DRY_RUN") == "True"
    assert mod.effective_param(nb, "NOT_A_PARAM") is None


def test_ci_executes_the_notebook_in_ablations_dry_run_mode() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--set CONFIG='\"ablations_l4\"' --set DRY_RUN=True" in ci
