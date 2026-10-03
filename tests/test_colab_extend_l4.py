from __future__ import annotations

# CONFIG = "extend_l4": the notebook runs ext_stable_l4 -> ext_branch_a_l4 -> ext_branch_b_l4 in one
# Colab session (PREREG 2026-10-02 rule 4). Like tests/test_colab_ablations_l4.py, the logic lives
# in notebook cells, so these tests exec the real cells by id with `run_step` replaced by a recorder
# that simulates what nmt.train leaves behind; nothing is ever trained.
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import nbformat
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "colab" / "train.ipynb"
PARAMS, URL, ABL_HELPERS, EXT_HELPERS = "c5459372", "f4b8d2a6", "a3c7e1b9", "e8b1d7a1"
ORDER = ("ext_stable_l4", "ext_branch_a_l4", "ext_branch_b_l4")
PLANNED = {"ext_stable_l4": 40000, "ext_branch_a_l4": 37500, "ext_branch_b_l4": 50000}


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
    values = {"CONFIG": '"extend_l4"', **edits}
    namespace: dict[str, Any] = {}
    replacements = {}
    for name, literal in values.items():
        line = next(ln for ln in _src(PARAMS).splitlines() if ln.startswith(f"{name} = "))
        replacements[line.split("  #")[0]] = f"{name} = {literal}"
    _exec(PARAMS, namespace, replacements)
    return namespace


class FakeRunStep:
    """Records every run_step call; a 'train' call writes run_info.json + its final checkpoint."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on = fail_on

    def __call__(self, step: str, argv: list[str], **kwargs: Any) -> str:
        cfg = Path(argv[argv.index("--config") + 1]).stem
        self.calls.append({"cfg": cfg, "argv": list(argv)})
        if cfg == self.fail_on:
            raise RuntimeError(f"{step} failed (exit 1)")
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        ckpt_dir = Path(argv[argv.index("--ckpt-dir") + 1])
        planned = PLANNED[cfg]
        steps = {30000, 40000, planned} if cfg == "ext_stable_l4" else {planned}
        for s in steps:
            (ckpt_dir / f"step_{s:08d}.pt").write_bytes(f"{cfg}:{s}".encode())
        info = {"exit_reason": "completed", "final_step": planned, "train_wall_seconds": 1.0}
        (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        return ""


def _ns(tmp_path: Path, run_step: Any, **param_edits: str) -> dict[str, Any]:
    ns = _params(**param_edits)
    ns.update({"Path": Path, "re": re, "run_step": run_step, "sys": sys})
    _exec(URL, ns)
    _exec(ABL_HELPERS, ns)
    ns["EXTEND"] = False  # the helper cell's bottom block (input check + estimate) is tested apart
    _exec(EXT_HELPERS, ns)
    return ns


def _dirs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path], Path]:
    run_dirs = {c: tmp_path / "runs" / f"notebook_{c}" for c in ORDER}
    return run_dirs, {c: d / "ckpt" for c, d in run_dirs.items()}, tmp_path / "runs" / "main_ckpt"


def _sequence(ns: dict[str, Any], tmp_path: Path, *, dry_run: bool = False) -> dict[str, str]:
    run_dirs, ckpt_dirs, main_ckpt = _dirs(tmp_path)
    return ns["run_extend_sequence"](
        ORDER,
        repo_dir=REPO_ROOT,
        run_dirs=run_dirs,
        ckpt_dirs=ckpt_dirs,
        main_ckpt_dir=main_ckpt,
        data_dir=tmp_path / "shards",
        wandb_mode="offline",
        preflight={"preflight_gpu": "NVIDIA L4"},
        dry_run=dry_run,
    )


def _put_main_input(tmp_path: Path) -> str:
    main_ckpt = _dirs(tmp_path)[2]
    main_ckpt.mkdir(parents=True)
    (main_ckpt / "step_00019000.pt").write_bytes(b"main-19000")
    return hashlib.sha256(b"main-19000").hexdigest()


# --- parameters -----------------------------------------------------------------------------


def test_extend_is_a_selectable_config_and_pins_the_new_tag() -> None:
    ns = _params(DRY_RUN="True")
    assert ns["EXTEND"] is True and ns["GIT_REF"] == "v0.3.1-colab"
    assert ns["PLANNED_STEPS"] is None  # ignored, not forwarded (24645 is main's value)
    assert ns["EXTEND_CONFIGS"] == ORDER
    assert ns["EXTEND_INIT"] == {
        "ext_stable_l4": ("main", 19000),
        "ext_branch_a_l4": ("ext_stable_l4", 30000),
        "ext_branch_b_l4": ("ext_stable_l4", 40000),
    }


@pytest.mark.parametrize("flag", ["COOLDOWN_NOW", "RESUME_TEST", "RUN_EVAL"])
def test_training_flags_are_refused(flag: str) -> None:
    with pytest.raises(ValueError, match=flag):
        _params(**{flag: "True"})


def test_extend_inits_match_the_config_milestones_and_files() -> None:
    cfgs = {n: yaml.safe_load((REPO_ROOT / "configs" / f"{n}.yaml").read_text()) for n in ORDER}
    assert {n: c["optim"]["planned_steps"] for n, c in cfgs.items()} == PLANNED
    assert {c["group"] for c in cfgs.values()} == {"extend_l4"}
    init = _params()["EXTEND_INIT"]
    stable = set(cfgs["ext_stable_l4"]["ckpt"]["milestone_steps"])
    assert init["ext_branch_a_l4"][1] in stable and init["ext_branch_b_l4"][1] in stable
    for name in ("ext_branch_a_l4", "ext_branch_b_l4"):  # branch decay starts at its input step
        assert cfgs[name]["optim"]["decay_start"] == init[name][1]


# --- estimate, input check ------------------------------------------------------------------


def test_estimate_is_labelled_and_matches_the_arithmetic(tmp_path: Path) -> None:
    ns = _ns(tmp_path, FakeRunStep())
    text = "\n".join(ns["extend_estimate_lines"]({"a": 21000, "b": 7500, "c": 10000}))
    assert (
        "ESTIMATE" in text and "0.4785 s/step" in text and "reports/main_l4/run_audit.json" in text
    )
    assert "21,000 + 7,500 + 10,000 = 38,500 steps" in text
    hours = 38500 * 0.4785 / 3600
    assert f"~{hours:.2f} h" in text and f"~{hours * 1.54:.2f} CU" in text


def test_input_report_prints_sha256_and_refuses_when_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ns = _ns(tmp_path, FakeRunStep())
    sha = _put_main_input(tmp_path)
    path = _dirs(tmp_path)[2] / "step_00019000.pt"
    ns["extend_input_report"]("ext_stable_l4", path, dry_run=False, produced_by_sequence=False)
    assert f"input sha256:     {sha}" in capsys.readouterr().out
    gone = path.with_name("step_00018999.pt")
    with pytest.raises(RuntimeError, match="MISSING"):
        ns["extend_input_report"]("ext_stable_l4", gone, dry_run=False, produced_by_sequence=False)
    ns["extend_input_report"]("ext_stable_l4", gone, dry_run=True, produced_by_sequence=False)
    assert "MISSING (a real run REFUSES" in capsys.readouterr().out


# --- sequencing -----------------------------------------------------------------------------


def test_missing_input_checkpoint_refuses_before_any_training(tmp_path: Path) -> None:
    fake = FakeRunStep()
    ns = _ns(tmp_path, fake)
    with pytest.raises(RuntimeError, match="step_00019000.pt is MISSING"):
        _sequence(ns, tmp_path)
    assert fake.calls == []  # nothing launched, nothing written
    assert not (tmp_path / "runs" / "notebook_ext_stable_l4").exists()


def test_sequence_runs_in_order_with_the_right_init_from(tmp_path: Path) -> None:
    _put_main_input(tmp_path)
    fake = FakeRunStep()
    ns = _ns(tmp_path, fake)
    decisions = _sequence(ns, tmp_path)
    assert decisions == dict.fromkeys(ORDER, "fresh")
    assert [c["cfg"] for c in fake.calls] == list(ORDER)
    _, ckpt_dirs, main_ckpt = _dirs(tmp_path)
    want = {
        "ext_stable_l4": main_ckpt / "step_00019000.pt",
        "ext_branch_a_l4": ckpt_dirs["ext_stable_l4"] / "step_00030000.pt",
        "ext_branch_b_l4": ckpt_dirs["ext_stable_l4"] / "step_00040000.pt",
    }
    for call in fake.calls:
        argv = call["argv"]
        assert argv[argv.index("--init-from") + 1] == str(want[call["cfg"]])
        assert "--resume" in argv and "--planned-steps" not in argv and "--cooldown-now" not in argv
        assert argv[argv.index("--ckpt-dir") + 1] == str(ckpt_dirs[call["cfg"]])
        assert argv[argv.index("--wandb") + 1] == "offline"
    # a second Run all: everything finished -> skipped, nothing relaunched
    again = FakeRunStep()
    ns2 = _ns(tmp_path, again)
    assert _sequence(ns2, tmp_path) == dict.fromkeys(ORDER, "finished")
    assert again.calls == []


def test_partial_run_resumes_without_init_from_and_later_runs_start_fresh(tmp_path: Path) -> None:
    _put_main_input(tmp_path)
    run_dirs, ckpt_dirs, _ = _dirs(tmp_path)
    ckpt_dirs["ext_stable_l4"].mkdir(parents=True)
    (ckpt_dirs["ext_stable_l4"] / "step_00027000.pt").write_bytes(b"x")  # interrupted mid-run
    fake = FakeRunStep()
    ns = _ns(tmp_path, fake)
    decisions = _sequence(ns, tmp_path)
    assert decisions["ext_stable_l4"] == "resume"
    first = fake.calls[0]["argv"]
    assert "--resume" in first and "--init-from" not in first  # own checkpoint wins
    assert all("--init-from" in c["argv"] for c in fake.calls[1:])


def test_first_failure_stops_the_sequence(tmp_path: Path) -> None:
    _put_main_input(tmp_path)
    fake = FakeRunStep(fail_on="ext_stable_l4")
    ns = _ns(tmp_path, fake)
    with pytest.raises(RuntimeError, match="failed"):
        _sequence(ns, tmp_path)
    assert [c["cfg"] for c in fake.calls] == ["ext_stable_l4"]  # branches never started


def test_dry_run_launches_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    fake = FakeRunStep()
    ns = _ns(tmp_path, fake)
    _sequence(ns, tmp_path, dry_run=True)  # input MISSING is reported, not fatal, in a dry run
    out = capsys.readouterr().out
    assert fake.calls == [] and not (tmp_path / "runs").exists()
    assert "EXTEND 1/3: ext_stable_l4" in out and "--init-from" in out
    assert "MISSING (a real run REFUSES" in out
    assert "MISSING (will be written by an earlier run of this sequence)" in out


# --- overfitting-watch lines ----------------------------------------------------------------


def test_watch_lines_report_flag_and_minimum(tmp_path: Path) -> None:
    ns = _ns(tmp_path, FakeRunStep())
    assert "no metrics.jsonl" in ns["extend_watch_lines"](tmp_path)[0]
    rows = [
        {"eval": {"step": 500, "val_loss": 2.0, "overfit_flag": 0}},
        {"eval": {"step": 1000, "val_loss": 1.8, "overfit_flag": 0}},
        {"eval": {"step": 1500, "val_loss": 1.9, "overfit_flag": 0}},
        {"eval": {"step": 2000, "val_loss": 1.95, "overfit_flag": 1, "val_loss_rises": 2}},
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    (line,) = ns["extend_watch_lines"](tmp_path)
    assert "4 evals" in line and "min 1.8000 (step 1000)" in line
    assert "FLAGGED since step 2000" in line


# --- wiring ---------------------------------------------------------------------------------


def test_ci_and_executor_cover_extend_l4_dry_run() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--set CONFIG='\"extend_l4\"' --set DRY_RUN=True" in ci
    executor = (REPO_ROOT / "colab" / "execute_notebook.py").read_text(encoding="utf-8")
    assert '"extend_l4"' in executor.split("DRY_RUN_CONFIGS = ")[1].splitlines()[0]


def test_expected_input_sha256_defaults_to_none_and_pins_when_set(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert _params()["EXPECTED_INPUT_SHA256"] is None  # never invented: GG pins it later
    sha = _put_main_input(tmp_path)
    path = _dirs(tmp_path)[2] / "step_00019000.pt"
    ns = _ns(tmp_path, FakeRunStep())
    report = ns["extend_input_report"]
    report("ext_stable_l4", path, dry_run=False, produced_by_sequence=False, expected_sha256=sha)
    assert "matches EXPECTED_INPUT_SHA256" in capsys.readouterr().out
    bad = "0" * 64
    with pytest.raises(RuntimeError, match="EXPECTED_INPUT_SHA256"):
        report(
            "ext_stable_l4", path, dry_run=False, produced_by_sequence=False, expected_sha256=bad
        )
    report("ext_stable_l4", path, dry_run=True, produced_by_sequence=False, expected_sha256=bad)
    assert "MISMATCH" in capsys.readouterr().out


def test_sequence_refuses_a_pin_mismatch_before_any_training(tmp_path: Path) -> None:
    _put_main_input(tmp_path)
    fake = FakeRunStep()
    ns = _ns(tmp_path, fake)
    ns["EXPECTED_INPUT_SHA256"] = "0" * 64
    with pytest.raises(RuntimeError, match="EXPECTED_INPUT_SHA256"):
        _sequence(ns, tmp_path)
    assert fake.calls == []


def test_helper_cell_checks_the_input_and_prints_the_estimate_before_training(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    sha = _put_main_input(tmp_path)
    _, ckpt_dirs, main_ckpt = _dirs(tmp_path)
    ns = _params(DRY_RUN="True")
    ns.update({"Path": Path, "re": re, "run_step": FakeRunStep(), "sys": sys})
    ns.update(repo_dir=REPO_ROOT, extend_ckpt_dirs=ckpt_dirs, main_ckpt_dir=main_ckpt)
    _exec(URL, ns)
    _exec(ABL_HELPERS, ns)
    capsys.readouterr()
    _exec(EXT_HELPERS, ns)  # EXTEND is True here: runs the bottom block
    out = capsys.readouterr().out
    assert f"input sha256:     {sha}" in out
    assert "21,000 + 7,500 + 10,000 = 38,500 steps" in out and "ESTIMATE" in out
