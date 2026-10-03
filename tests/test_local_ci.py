from __future__ import annotations

# Offline tests for scripts/local_ci.py: the ci.yml <-> STEPS drift guard (the main one), the
# pure helpers, the cleanup safety checks on a throwaway git repo, and a --dry end-to-end pass.
import subprocess
from pathlib import Path

import pytest

from scripts import local_ci

REPO = Path(__file__).resolve().parents[1]
CI_YML = REPO / ".github" / "workflows" / "ci.yml"


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def tiny_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_every_ci_yml_run_command_is_in_the_table() -> None:
    """ci.yml and the local table must not drift: same steps, same run text, same order."""
    assert local_ci.ci_drift(CI_YML.read_text(encoding="utf-8")) == []


def test_every_run_block_is_literally_covered() -> None:
    wf = local_ci.workflow_steps(CI_YML.read_text(encoding="utf-8"))
    runs = [local_ci.norm(str(w["run"])) for w in wf if w["run"]]
    table = {local_ci.norm(s.ci_run) for s in local_ci.STEPS if s.ci_run}
    assert runs, "ci.yml has no run: steps?"
    assert set(runs) <= table


def test_drift_is_detected_when_a_command_changes() -> None:
    text = CI_YML.read_text(encoding="utf-8").replace("uv run pytest -q", "uv run pytest -x -q", 1)
    problems = local_ci.ci_drift(text)
    assert any("Test (pytest)" in p for p in problems)


def test_drift_is_detected_when_a_step_is_added() -> None:
    text = CI_YML.read_text(encoding="utf-8").rstrip("\n") + (
        "\n      - name: Brand new step\n        run: echo hi\n"
    )
    problems = local_ci.ci_drift(text)
    assert any("Brand new step" in p for p in problems)


def test_unsupported_if_condition_fails_closed() -> None:
    text = CI_YML.read_text(encoding="utf-8").replace(
        "if: matrix.python-version == '3.12'", "if: github.ref == 'x'", 1
    )
    with pytest.raises(local_ci.CiError):
        local_ci.workflow_steps(text)


def test_handlers_cover_every_step() -> None:
    assert {s.handler for s in local_ci.STEPS} <= set(local_ci.HANDLERS)


def test_torch_version_from_pyproject_matches_the_sed() -> None:
    assert local_ci.torch_version_from_pyproject('x = [\n    "torch==2.14.0",\n]\n') == "2.14.0"
    assert local_ci.torch_version_from_pyproject('"torch>=2",\n') == ""
    real = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert local_ci.torch_version_from_pyproject(real) != ""


def test_parse_pytest_summary_quiet_and_verbose() -> None:
    quiet = ["....", "SKIPPED [1] tests/t.py:3: needs cuda", "12 passed, 2 skipped in 3.20s"]
    out = local_ci.parse_pytest_summary(quiet)
    assert out["counts"] == {"passed": 12, "skipped": 2}
    assert out["skip_reasons"] == ["SKIPPED [1] tests/t.py:3: needs cuda"]
    verbose = ["===== 5 passed, 1 failed, 1 error in 0.50s ====="]
    assert local_ci.parse_pytest_summary(verbose)["counts"] == {
        "passed": 5,
        "failed": 1,
        "errors": 1,
    }
    assert local_ci.parse_pytest_summary(["no summary"])["counts"] == {}


def test_resolve_ref(tiny_repo: tuple[Path, str]) -> None:
    repo, sha = tiny_repo
    assert local_ci.resolve_ref(repo, "HEAD") == sha
    assert local_ci.resolve_ref(repo, sha[:8]) == sha
    with pytest.raises(local_ci.CiError):
        local_ci.resolve_ref(repo, "no-such-ref")


def test_env_cache_key_tracks_lock_content(tmp_path: Path) -> None:
    (tmp_path / "colab").mkdir()
    for rel in ("uv.lock", "pyproject.toml", "requirements-colab.txt"):
        (tmp_path / rel).write_text("x", encoding="utf-8")
    (tmp_path / "colab" / "make_torch_constraints.py").write_text("y", encoding="utf-8")
    k1 = local_ci.env_cache_key(tmp_path, "3.12", "Python 3.12.12")
    assert k1 == local_ci.env_cache_key(tmp_path, "3.12", "Python 3.12.12")
    assert k1 != local_ci.env_cache_key(tmp_path, "3.12", "Python 3.12.13")
    (tmp_path / "uv.lock").write_text("changed", encoding="utf-8")
    assert k1 != local_ci.env_cache_key(tmp_path, "3.12", "Python 3.12.12")
    k13 = local_ci.env_cache_key(tmp_path, "3.13", "Python 3.13.1")
    (tmp_path / "uv.lock").write_text("changed again", encoding="utf-8")
    assert k13 == local_ci.env_cache_key(tmp_path, "3.13", "Python 3.13.1")  # lock not an input


def test_cleanup_check_then_delete(tiny_repo: tuple[Path, str], tmp_path: Path) -> None:
    repo, sha = tiny_repo
    work_root = tmp_path / "ci_work"
    wt = work_root / sha / "py312"
    wt.parent.mkdir(parents=True)
    _git(repo, "worktree", "add", "--detach", str(wt), sha)
    plan = local_ci.plan_cleanup(repo, work_root, sha, [wt])
    assert plan.problems == []
    assert wt.exists()  # the check phase deletes nothing
    local_ci.execute_cleanup(repo, plan)
    assert not (work_root / sha).exists()
    assert (repo / "a.txt").exists()  # the user's repo is untouched


def test_cleanup_refuses_paths_outside_ci_work(tiny_repo: tuple[Path, str], tmp_path: Path) -> None:
    repo, sha = tiny_repo
    work_root = tmp_path / "ci_work"
    work_root.mkdir()
    outside = tmp_path / "elsewhere" / "py312"
    outside.parent.mkdir()
    _git(repo, "worktree", "add", "--detach", str(outside), sha)
    plan = local_ci.plan_cleanup(repo, work_root, sha, [outside])
    assert plan.problems
    with pytest.raises(local_ci.CiError):
        local_ci.execute_cleanup(repo, plan)
    assert outside.exists()


def test_cleanup_refuses_unregistered_directory(
    tiny_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, sha = tiny_repo
    work_root = tmp_path / "ci_work"
    fake = work_root / sha / "py312"
    fake.mkdir(parents=True)
    (fake / "precious.txt").write_text("user data", encoding="utf-8")
    plan = local_ci.plan_cleanup(repo, work_root, sha, [fake])
    assert any("not a registered worktree" in p for p in plan.problems)
    with pytest.raises(local_ci.CiError):
        local_ci.execute_cleanup(repo, plan)
    assert (fake / "precious.txt").exists()


def test_cleanup_refuses_non_sha_name(tiny_repo: tuple[Path, str], tmp_path: Path) -> None:
    repo, _ = tiny_repo
    plan = local_ci.plan_cleanup(repo, tmp_path / "ci_work", "../..", [])
    assert plan.problems


def test_list_steps_prints_every_ci_step(capsys: pytest.CaptureFixture[str]) -> None:
    assert local_ci.main(["--list-steps"]) == 0
    out = capsys.readouterr().out
    for s in local_ci.STEPS:
        assert s.name in out


def test_dry_run_prints_every_command_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FR_EN_CI_ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(local_ci.shutil, "which", lambda _name: "uv")
    monkeypatch.setattr(local_ci, "_interpreter_version_uv", lambda _uv, spec: f"Python {spec}")
    monkeypatch.setattr(local_ci, "environment_info", lambda _uv: {})
    rc = local_ci.main(["--ref", "HEAD", "--dry"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "$ uv sync --frozen --python 3.12" in out
    assert '--set CONFIG="eval_l4" --set DRY_RUN=True --set RUN="final_all"' in out
    assert "PASS" in out
    assert not (tmp_path / "root").exists()
