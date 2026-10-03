from __future__ import annotations

# Offline tests for scripts/local_ci.py: the ci.yml <-> STEPS drift guard (the main one), the
# pure helpers, the cleanup safety checks on a throwaway git repo, and a --dry end-to-end pass.
import copy
import os
import subprocess
import sys
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
    assert rc == 3, out  # distinct from PASS (0) and FAIL (1)
    assert "$ uv sync --frozen --python 3.12" in out
    assert '--set CONFIG="eval_l4" --set DRY_RUN=True --set RUN="final_all"' in out
    assert "DRY RUN (nothing executed)" in out
    assert "PASS" not in out
    assert not (tmp_path / "root").exists()


# ---------------------------------------------------------------------------------------------
# helpers for the link / worktree tests
# ---------------------------------------------------------------------------------------------
def _make_link(link: Path, target: Path) -> None:
    """Directory junction on Windows (what bit us), symlink elsewhere."""
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
    else:
        os.symlink(target, link, target_is_directory=True)


def _worktree(repo: Path, sha: str, work_root: Path, name: str = "py312") -> Path:
    wt = work_root / sha / name
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(wt), sha)
    return wt


def _precious(tmp_path: Path) -> Path:
    d = tmp_path / "precious"
    d.mkdir()
    (d / "keep.txt").write_text("irreplaceable", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    return d


def _assert_intact(d: Path) -> None:
    assert (d / "keep.txt").read_text(encoding="utf-8") == "irreplaceable"
    assert (d / "sub" / "deep.txt").read_text(encoding="utf-8") == "deep"


# ---------------------------------------------------------------------------------------------
# (1) junction-safe cleanup
# ---------------------------------------------------------------------------------------------
def test_cleanup_refuses_link_pointing_outside_ci_work(
    tiny_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, sha = tiny_repo
    work_root = tmp_path / "ci_work"
    wt = _worktree(repo, sha, work_root)
    precious = _precious(tmp_path)
    _make_link(wt / "data_link", precious)
    plan = local_ci.plan_cleanup(repo, work_root, sha, [wt])
    assert any("points outside ci_work" in p for p in plan.problems)
    with pytest.raises(local_ci.CiError):
        local_ci.execute_cleanup(repo, plan)
    _assert_intact(precious)
    assert wt.exists()


def test_cleanup_unlinks_inside_link_without_following_it(
    tiny_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, sha = tiny_repo
    work_root = tmp_path / "ci_work"
    wt = _worktree(repo, sha, work_root)
    inside = work_root / sha / "scratch_target"
    inside.mkdir()
    (inside / "keep.txt").write_text("irreplaceable", encoding="utf-8")
    (inside / "sub").mkdir()
    (inside / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    _make_link(wt / "junc", inside)
    plan = local_ci.plan_cleanup(repo, work_root, sha, [wt])
    assert plan.problems == []
    assert plan.links == [wt / "junc"]
    local_ci.execute_cleanup(repo, plan)
    assert not (work_root / sha).exists()  # whole sha dir gone, nothing followed


def test_cleanup_force_path_never_follows_a_link_to_a_precious_dir(
    tiny_repo: tuple[Path, str], tmp_path: Path
) -> None:
    """Regression for `git worktree remove --force` following an inner junction: even if a link
    to an outside dir appears AFTER the check phase, the delete phase must not touch the target."""
    repo, sha = tiny_repo
    work_root = tmp_path / "ci_work"
    wt = _worktree(repo, sha, work_root)
    precious = _precious(tmp_path)
    (wt / "untracked.txt").write_text("makes plain `worktree remove` refuse", encoding="utf-8")
    plan = local_ci.plan_cleanup(repo, work_root, sha, [wt])
    assert plan.problems == []
    _make_link(wt / "late_link", precious)  # appears between check and delete
    # the delete phase unlinks it first (never follows), so the target survives
    local_ci.execute_cleanup(repo, plan)
    _assert_intact(precious)
    assert not (work_root / sha).exists()


def test_find_links_does_not_descend_into_links(tmp_path: Path) -> None:
    precious = _precious(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    _make_link(root / "j", precious)
    assert local_ci.find_links(root) == [root / "j"]


# ---------------------------------------------------------------------------------------------
# (2) drift guard compares everything in ci.yml
# ---------------------------------------------------------------------------------------------
def _ci_doc() -> dict:
    import yaml

    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


def _dump(doc: dict) -> str:
    import yaml

    return yaml.safe_dump(doc, sort_keys=False)


def _job(doc: dict) -> dict:
    return doc["jobs"]["lint-and-test"]


def _step(doc: dict, name: str) -> dict:
    return next(s for s in _job(doc)["steps"] if s.get("name") == name)


def test_roundtripped_ci_yml_has_no_drift() -> None:
    """Control for the mutation tests below: dump/parse alone changes nothing."""
    assert local_ci.ci_drift(_dump(_ci_doc())) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: _step(d, "Lint (ruff check)").__setitem__("env", {"X": "1"}),
        lambda d: _step(d, "Lint (ruff check)").__setitem__("continue-on-error", True),
        lambda d: _step(d, "Lint (ruff check)").__setitem__("timeout-minutes", 1),
        lambda d: _step(d, "Lint (ruff check)").__setitem__("shell", "bash"),
        lambda d: _step(d, "Lint (ruff check)").__setitem__("working-directory", "x"),
        lambda d: _step(d, "Scorer-encoding tests, verbose (must run, not skip)").__setitem__(
            "shell", "pwsh"
        ),
        lambda d: _step(d, "Set up Python 3.13")["with"].__setitem__("python-version", "3.14"),
        lambda d: _step(d, "Test (pytest)").__setitem__("if", "matrix.python-version == '3.13'"),
        lambda d: _job(d)["strategy"]["matrix"]["python-version"].append("3.14"),
        lambda d: _job(d)["strategy"].__setitem__("fail-fast", True),
        lambda d: _job(d).__setitem__("env", {"UV_NO_SYNC": "1"}),
        lambda d: _job(d).__setitem__("runs-on", "windows-latest"),
        lambda d: _job(d).__setitem__("continue-on-error", True),
        lambda d: _job(d).__setitem__("timeout-minutes", 5),
        lambda d: d.__setitem__("env", {"CI_FLAG": "1"}),
        lambda d: d["concurrency"].__setitem__("cancel-in-progress", False),
        lambda d: d.__setitem__("permissions", {"contents": "write"}),
        lambda d: d.__setitem__("jobs", {**d["jobs"], "extra": copy.deepcopy(_job(d))}),
    ],
)
def test_any_ci_yml_change_is_reported_as_drift(mutate) -> None:
    doc = copy.deepcopy(_ci_doc())
    mutate(doc)
    assert local_ci.ci_drift(_dump(doc)) != []


# ---------------------------------------------------------------------------------------------
# (4) git failure is reported, (5) cleanup in finally + prune, (6) lock
# ---------------------------------------------------------------------------------------------
def test_prepare_job_reports_git_worktree_add_failure(
    tiny_repo: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = tiny_repo
    monkeypatch.setattr(local_ci, "_interpreter_version_uv", lambda _uv, spec: f"Python {spec}")
    with pytest.raises(local_ci.CiError, match="worktree add"):
        local_ci.prepare_job(repo, tmp_path / "root", "0" * 40, "3.12", "uv", dry=False, echo=False)


def _patch_main(monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path) -> Path:
    root = tmp_path / "root"
    monkeypatch.setenv("FR_EN_CI_ROOT", str(root))
    monkeypatch.setattr(local_ci, "REPO_ROOT", repo)
    monkeypatch.setattr(local_ci.shutil, "which", lambda _name: "uv")
    monkeypatch.setattr(local_ci, "_interpreter_version_uv", lambda _uv, spec: f"Python {spec}")
    return root


def test_worktrees_are_cleaned_up_when_a_job_raises(
    tiny_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, sha = tiny_repo
    root = _patch_main(monkeypatch, repo, tmp_path)

    def boom(_ctx: object) -> dict:
        raise local_ci.CiError("synthetic failure")

    monkeypatch.setattr(local_ci, "env_cache_key", lambda *_a: "k")
    monkeypatch.setattr(local_ci, "run_job", boom)
    rc = local_ci.main(["--ref", sha, "--allow-ci-drift", "--quiet"])
    assert rc == 2
    assert "synthetic failure" in capsys.readouterr().err
    assert not (root / "ci_work" / sha).exists()
    assert "ci_work" not in _git(repo, "worktree", "list")


def test_prune_lists_without_deleting_then_removes_with_yes(
    tiny_repo: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, sha = tiny_repo
    root = _patch_main(monkeypatch, repo, tmp_path)
    wt = _worktree(repo, sha, root / "ci_work")
    other = tmp_path / "unrelated_worktree"  # a user's own worktree must never be listed
    _git(repo, "worktree", "add", "--detach", str(other), sha)
    assert local_ci.main(["--prune"]) == 0
    out = capsys.readouterr().out
    assert sha in out and "listing only" in out and str(other) not in out
    assert wt.exists()
    assert local_ci.main(["--prune", "--yes"]) == 0
    assert not (root / "ci_work" / sha).exists()
    assert other.exists()


def test_prune_yes_refuses_a_link_to_outside(
    tiny_repo: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha = tiny_repo
    root = _patch_main(monkeypatch, repo, tmp_path)
    wt = _worktree(repo, sha, root / "ci_work")
    precious = _precious(tmp_path)
    _make_link(wt / "j", precious)
    assert local_ci.main(["--prune", "--yes"]) == 1
    _assert_intact(precious)
    assert wt.exists()


def test_env_lock_excludes_a_second_holder_and_releases(tmp_path: Path) -> None:
    lock = tmp_path / "ci_envs" / "py312-x.lock"
    with local_ci.env_lock(lock, wait_s=0):
        assert lock.exists()
        with (
            pytest.raises(local_ci.CiError, match="held by"),
            local_ci.env_lock(lock, wait_s=0, poll_s=0.01),
        ):
            pass
        assert lock.exists()  # the failed contender must not release the holder's lock
    assert not lock.exists()
    with local_ci.env_lock(lock, wait_s=0):  # reusable after release
        pass


def test_env_lock_released_when_body_raises(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    with pytest.raises(RuntimeError), local_ci.env_lock(lock, wait_s=0):
        raise RuntimeError("boom")
    assert not lock.exists()


def test_env_lock_takes_over_a_dead_pids_lock(tmp_path: Path) -> None:
    dead = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    lock = tmp_path / "x.lock"
    lock.write_text(f"{dead.stdout.strip()} 0\n", encoding="utf-8")
    assert not local_ci.pid_alive(int(dead.stdout.strip()))
    with local_ci.env_lock(lock, wait_s=0):
        assert lock.read_text(encoding="utf-8").split()[0] == str(os.getpid())


def test_env_lock_respects_a_live_holder(tmp_path: Path) -> None:
    import time

    lock = tmp_path / "x.lock"
    lock.write_text(f"{os.getpid()} {time.time()}\n", encoding="utf-8")
    assert local_ci.pid_alive(os.getpid())
    with pytest.raises(local_ci.CiError), local_ci.env_lock(lock, wait_s=0):
        pass
