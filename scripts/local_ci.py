from __future__ import annotations

# Local CI runner: reproduces .github/workflows/ci.yml step for step on this machine, because
# GitHub Actions is unavailable for the private repo (billing). The step list is the declarative
# STEPS table below, mirroring ci.yml's step names one-to-one; each entry carries the verbatim
# `run:` text of the workflow step (CI_RUN) and a Python handler that is its explicit Windows
# translation. tests/test_local_ci.py parses ci.yml and fails if the two drift, and every run
# re-checks the ci.yml of the SHA under test (fail closed; --allow-ci-drift downgrades to a warning
# recorded in the summary).
#
# Usage (from the repo root, any Python >= 3.12 with PyYAML, e.g. `uv run python -m ...`):
#   python -m scripts.local_ci [--ref <sha|branch>] [--jobs 3.12,3.13] [--keep] [--parallel]
#                              [--list-steps] [--dry] [--allow-ci-drift] [--quiet]
#
# What a run does: resolves --ref (default HEAD) to a full SHA; makes one CLEAN detached git
# worktree per job under <root>/ci_work/<sha>/ (the user's working trees are never touched); builds
# or reuses a cached environment per job under <root>/ci_envs/ (key = hash of the lock/requirement
# files + interpreter; the CI install steps are re-run on every use, so the env is always verified
# against the lock); runs the job's steps in order, stopping a job at its first failing step (the
# other job still runs, like fail-fast: false); writes <root>/ci_logs/<sha>__<job>.log and
# <root>/ci_logs/<sha>.json; prints a PASS/FAIL line naming the first failing step per job; then
# removes its own temp checkouts, as a SEPARATE check phase (plan_cleanup) and delete phase
# (execute_cleanup); it never deletes anything outside <root>/ci_work. The root is
# D:\ml-runs\fr-en-transformer, override with the FR_EN_CI_ROOT environment variable.
# Exit code: 0 = every job PASS, 1 = any step failed, 2 = could not run (bad ref, no uv, ...).
#
# KNOWN DEVIATIONS from ci.yml (also listed in the PR body):
#   1. Runs on Windows, not ubuntu-latest: bash steps are translated to argv + Python checks
#      (grep/tee/sed equivalents), "$RUNNER_TEMP" is a per-job dir under ci_work/<sha>/, and the
#      3.13 venv python is <venv>\Scripts\python.exe instead of <venv>/bin/python.
#   2. actions/checkout -> `git worktree add --detach` of the exact SHA, one per job (as Actions
#      gives each matrix job its own checkout). actions/setup-uv -> uv must already be on PATH.
#      setup-python 3.13 / `uv python install 3.12` -> uv-managed interpreters
#      (`uv python install` is a no-op when present; it downloads only if missing).
#   3. Cached environments (ci_envs/): ci.yml starts from an empty runner. The install steps still
#      run every time (uv sync --frozen exact-syncs; pip re-resolves pins against the installed
#      set), so the env is verified, not trusted. Local-only steps "[local] ..." are ADDED (env
#      verification, freeze hash); they never replace a ci.yml step.
#   4. PYTEST_ADDOPTS=-rs is set on the `pytest -q` steps so skip reasons are recorded in the
#      summary; it changes reporting only, not test selection.
#  5a. The final_all DRY_RUN step (bash: `set -eo pipefail`, `... | tee log`, a `for needle in ...;
#      grep -qF` loop over the COMET section) is translated to: run the notebook, capture its output
#      (ctx.run) and assert every needle of FINAL_ALL_DRY_RUN_NEEDLES with Python; the needle list
#      is compared with ci.yml's loop by tests/test_local_ci.py. A needle missing, or a non-zero
#      notebook exit, fails the step exactly as the bash step does.
#   5. Provider-token variables (HUGGINGFACEHUB_API_TOKEN, HF_TOKEN, ...) are removed from the
#      step environment, as a GitHub runner has none, plus VIRTUAL_ENV / PYTHONPATH.
#   6. Per-step timeout of STEP_TIMEOUT_S (the notebooks carry their own 1800 s timeout).
#   7. Python patch versions come from the local uv-managed interpreters, not setup-python's.
#   8. HEAD SHA vs MERGE REF: on pull_request Actions tests the synthetic merge commit
#      (refs/pull/N/merge = PR head merged into current main); this runner tests exactly the
#      commit given by --ref. To reproduce the Actions view, pass a SHA of the merge result
#      (e.g. `git merge --no-commit` in a scratch branch, or the PR's merge ref) in addition to
#      the head SHA. A head that passes here can still fail after merging into a moved main.
#   9. uv VERSION IS UNPINNED: Actions uses astral-sh/setup-uv@v10.2.0; this runner uses whatever
#      `uv` is on PATH (recorded in the summary under environment.uv). uv.lock is frozen, so
#      resolved packages do not change, but uv's own behaviour can differ between versions.
# Added safety/robustness (not ci.yml deviations): cleanup never follows junctions/symlinks (they
# are unlinked first and refused if they point outside ci_work), cleanup runs in a `finally`,
# per-env lockfiles serialise concurrent runs, `--prune [--yes]` lists/removes stale ci_work
# worktrees, `--dry` exits 3 and prints no PASS.
# Exit codes: 0 PASS, 1 FAIL, 2 could not run, 3 dry run (nothing executed).
import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(r"D:\ml-runs\fr-en-transformer")
JOBS_ALL = ("3.12", "3.13")
STEP_TIMEOUT_S = 3600
PIP_PIN = "26.2.1"  # mirrors `pip==26.2.1` in the ci.yml install step
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# Provider credentials a GitHub runner does not have; a stale/invalid token must not leak in.
SCRUBBED_ENV = (
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "UV_PROJECT_ENVIRONMENT",
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class CiError(Exception):
    """A condition that stops the runner itself (exit 2), not a failing CI step."""


# --------------------------------------------------------------------------------------------
# The step table. `ci_run` is the verbatim `run:` text from ci.yml (whitespace-normalised on
# comparison); `ci_uses` is set instead for `uses:` steps. `handler` names a function in
# HANDLERS. local_only steps are additions that have no ci.yml counterpart.
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Step:
    job: str  # "3.12", "3.13" or "*" (both jobs)
    name: str  # ci.yml step name (or the `uses:` string for unnamed steps)
    handler: str
    ci_run: str | None = None
    ci_uses: str | None = None
    local_only: bool = False
    # Every other key of the ci.yml step (with/shell/env/...), compared exactly by ci_drift.
    ci_extra: dict[str, object] = dataclasses.field(default_factory=dict)


_RUN_SCORER = """\
set -eo pipefail
if [ ! -f official/score.py ]; then
  echo "official/score.py absent (public checkout): scorer-encoding step not applicable"
  exit 0
fi
uv run pytest tests/test_scorer_encoding.py tests/test_official_scorer.py -v -rs \\
  | tee "$RUNNER_TEMP/scorer_encoding.log"
if grep -q " SKIPPED" "$RUNNER_TEMP/scorer_encoding.log"; then
  echo "scorer-encoding tests were SKIPPED; they must run" >&2
  exit 1
fi
"""

_RUN_INSTALL_313 = """\
python -m venv "$RUNNER_TEMP/venv"
PY="$RUNNER_TEMP/venv/bin/python"
"$PY" -m pip install --disable-pip-version-check pip==26.2.1
# Stand-in for Colab's preinstalled torch: the CPU wheel of the version pinned in
# pyproject.toml (the same version uv.lock resolves for CPU/CI).
TORCH_VERSION=$(sed -n 's/^ *"torch==\\([0-9.]*\\)",$/\\1/p' pyproject.toml)
test -n "$TORCH_VERSION"
"$PY" -m pip install --disable-pip-version-check "torch==${TORCH_VERSION}" \\
  --index-url https://download.pytorch.org/whl/cpu
# Same helper as the notebook's install cell, so the two cannot drift.
"$PY" colab/make_torch_constraints.py "$RUNNER_TEMP/torch_constraints.txt"
"$PY" -m pip install --disable-pip-version-check \\
  -r requirements-colab.txt -c "$RUNNER_TEMP/torch_constraints.txt"
"$PY" -m pip install --disable-pip-version-check -e . --no-deps
# Test + notebook-execution tooling, pinned to the versions in the uv dev group.
"$PY" -m pip install --disable-pip-version-check \\
  pytest==9.1.1 nbclient==0.11.0 nbformat==5.11.1 ipykernel==7.4.0
# pip must not have swapped the preinstalled torch.
"$PY" -c "import torch; print('torch', torch.__version__)"
grep -qx "torch==$("$PY" -c 'import torch; print(torch.__version__)')" \\
  "$RUNNER_TEMP/torch_constraints.txt"
"$PY" -m pip check
"""

_RUN_PYTEST_313 = """\
"$RUNNER_TEMP/venv/bin/python" -m pytest -q
"""

_RUN_SMOKE = """\
PY="$RUNNER_TEMP/venv/bin/python"
"$PY" colab/make_synthetic_shards.py .
# The notebook's defaults are the real L4 main run; say "smoke" explicitly here.
"$PY" colab/execute_notebook.py colab/train.ipynb 1800 \\
  --set CONFIG='"smoke"' --set PLANNED_STEPS=None --set RESUME_TEST=False
"""

_RUN_ABL = """\
PY="$RUNNER_TEMP/venv/bin/python"
"$PY" colab/execute_notebook.py colab/train.ipynb 1800 \\
  --set CONFIG='"ablations_l4"' --set DRY_RUN=True
"""

_RUN_EVAL_ALL = """\
PY="$RUNNER_TEMP/venv/bin/python"
"$PY" colab/execute_notebook.py colab/train.ipynb 1800 \\
  --set CONFIG='"eval_l4"' --set DRY_RUN=True --set RUN='"all"'
"""

# The COMET section the final_all dry run must print (ci.yml's `for needle in` loop). ONE copy
# here; tests/test_local_ci.py parses the loop out of ci.yml and compares it with this tuple.
FINAL_ALL_DRY_RUN_NEEDLES: tuple[str, ...] = (
    "COMET stage: after the upload, NON-FATAL",
    "[comet:install]",
    "[comet:pull:main]",
    "[comet:score]",
    "[comet:upload]",
    "61 sets",
    "Unbabel/wmt22-comet-da @",
    "batch size 64",
    "17294 distinct (exact",
    "4385 distinct (exact",
    "ASSUMED 50 triples/s",
    "ASSUMED 150 triples/s",
    "NO L4 COMET rate has been measured",
)

_RUN_FINAL_ALL = """\
set -eo pipefail
PY="$RUNNER_TEMP/venv/bin/python"
"$PY" colab/execute_notebook.py colab/train.ipynb 1800 \\
  --set CONFIG='"eval_l4"' --set DRY_RUN=True --set RUN='"final_all"' \\
  | tee "$RUNNER_TEMP/final_all_dry_run.log"
for needle in \\
  "COMET stage: after the upload, NON-FATAL" \\
  "[comet:install]" "[comet:pull:main]" "[comet:score]" "[comet:upload]" \\
  "61 sets" "Unbabel/wmt22-comet-da @" "batch size 64" \\
  "17294 distinct (exact" "4385 distinct (exact" \\
  "ASSUMED 50 triples/s" "ASSUMED 150 triples/s" \\
  "NO L4 COMET rate has been measured"; do
  grep -qF -- "$needle" "$RUNNER_TEMP/final_all_dry_run.log" \\
    || { echo "final_all dry run lacks: $needle" >&2; exit 1; }
done
"""

_RUN_EXTEND = """\
PY="$RUNNER_TEMP/venv/bin/python"
"$PY" colab/execute_notebook.py colab/train.ipynb 1800 \\
  --set CONFIG='"extend_l4"' --set DRY_RUN=True
"""

STEPS: tuple[Step, ...] = (
    Step("*", "actions/checkout@v4", "checkout", ci_uses="actions/checkout@v4"),
    Step("3.12", "Install uv", "install_uv", ci_uses="astral-sh/setup-uv@v10.2.0"),
    Step("3.12", "Install Python 3.12", "uv_python_312", ci_run="uv python install 3.12"),
    Step("3.12", "Check uv.lock is up to date", "uv_lock_check", ci_run="uv lock --check"),
    Step(
        "3.12",
        "Sync dependencies (frozen)",
        "uv_sync",
        ci_run="uv sync --frozen --python 3.12",
    ),
    Step("3.12", "Lint (ruff check)", "ruff_check", ci_run="uv run ruff check ."),
    Step(
        "3.12",
        "Format check (ruff format)",
        "ruff_format",
        ci_run="uv run ruff format --check .",
    ),
    Step("3.12", "Test (pytest)", "pytest_312", ci_run="uv run pytest -q"),
    Step(
        "3.12",
        "Scorer-encoding tests, verbose (must run, not skip)",
        "scorer_encoding",
        ci_run=_RUN_SCORER,
        ci_extra={"shell": "bash"},
    ),
    Step("3.12", "[local] verify env + checkout", "local_verify_312", local_only=True),
    Step(
        "3.13",
        "Set up Python 3.13",
        "setup_python_313",
        ci_uses="actions/setup-python@v5",
        ci_extra={"with": {"python-version": "3.13"}},
    ),
    Step(
        "3.13",
        "Create venv and install the Colab-style environment",
        "install_313",
        ci_run=_RUN_INSTALL_313,
    ),
    Step("3.13", "Test (pytest, Python 3.13)", "pytest_313", ci_run=_RUN_PYTEST_313),
    Step(
        "3.13",
        "Execute colab/train.ipynb in smoke mode (synthetic shards, Python 3.13)",
        "nb_smoke",
        ci_run=_RUN_SMOKE,
    ),
    Step(
        "3.13",
        "Execute colab/train.ipynb in ablations_l4 DRY_RUN mode (synthetic shards, 3.13)",
        "nb_ablations",
        ci_run=_RUN_ABL,
    ),
    Step(
        "3.13",
        "Execute colab/train.ipynb in eval_l4 RUN=all DRY_RUN mode (Python 3.13)",
        "nb_eval_all",
        ci_run=_RUN_EVAL_ALL,
    ),
    Step(
        "3.13",
        "Execute colab/train.ipynb in eval_l4 RUN=final_all DRY_RUN mode (Python 3.13)",
        "nb_final_all",
        ci_run=_RUN_FINAL_ALL,
        ci_extra={"shell": "bash"},
    ),
    Step(
        "3.13",
        "Execute colab/train.ipynb in extend_l4 DRY_RUN mode (synthetic shards, 3.13)",
        "nb_extend",
        ci_run=_RUN_EXTEND,
    ),
    Step("3.13", "[local] verify env + checkout", "local_verify_313", local_only=True),
)


def norm(text: str) -> str:
    """Whitespace-normalise a run block (ci.yml indentation/continuations are not semantic)."""
    return " ".join(text.split())


# --------------------------------------------------------------------------------------------
# ci.yml <-> STEPS drift check (used by the test and at run time on the SHA's own ci.yml)
# --------------------------------------------------------------------------------------------
# Everything in ci.yml outside the step list, compared exactly (name, `on`, concurrency, the job's
# runs-on / strategy / matrix / env / timeout ...). yaml 1.1 parses the key `on` as True; the
# expected side goes through the same parser so both sides agree.
CI_META_YAML = """\
name: CI
on:
  push:
  pull_request:
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.12", "3.13"]
    steps: []
"""


def _load_yaml(text: str) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CiError("PyYAML is required (it is a project dependency): use `uv run`") from exc
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "jobs" not in doc:
        raise CiError("ci.yml is not a workflow mapping with `jobs`")
    return doc


def _meta(doc: dict) -> dict:
    """The workflow minus every job's `steps` list."""
    out = {k: v for k, v in doc.items() if k != "jobs"}
    out["jobs"] = {
        name: {k: v for k, v in job.items() if k != "steps"} for name, job in doc["jobs"].items()
    }
    return out


def _gate(step: dict) -> str:
    """Matrix python-version a step is gated on via `if:` ("*" if ungated); fail closed."""
    cond = str(step.get("if", ""))
    m = re.fullmatch(r"matrix\.python-version == '([0-9.]+)'", cond)
    if cond and not m:
        raise CiError(f"unsupported `if:` {cond!r}; extend local_ci.py")
    return m.group(1) if m else "*"


def workflow_steps(ci_yml_text: str) -> list[dict[str, object]]:
    """[{job, name, run, uses, raw}] for every step of every ci.yml job, in file order.

    `raw` is the complete parsed step mapping (all keys). Raises CiError on a shape this runner
    does not understand.
    """
    doc = _load_yaml(ci_yml_text)
    out: list[dict[str, object]] = []
    for job in doc["jobs"].values():
        for step in job["steps"]:
            out.append(
                {
                    "job": _gate(step),
                    "name": step.get("name") or step.get("uses"),
                    "run": step.get("run"),
                    "uses": step.get("uses"),
                    "raw": step,
                }
            )
    return out


def expected_step_dict(s: Step) -> dict[str, object]:
    """The complete ci.yml mapping (minus `run`) a table entry stands for."""
    d: dict[str, object] = {}
    if s.name != s.ci_uses:
        d["name"] = s.name
    if s.job != "*":
        d["if"] = f"matrix.python-version == '{s.job}'"
    if s.ci_uses is not None:
        d["uses"] = s.ci_uses
    d.update(s.ci_extra)
    return d


def ci_drift(ci_yml_text: str, steps: Sequence[Step] = STEPS) -> list[str]:
    """Human-readable differences between ci.yml and the STEPS table ([] = in sync).

    Compares the FULL parsed step mapping (every key: env, with, shell, continue-on-error,
    timeout-minutes, ...), the run text, and everything outside the steps (job keys, matrix,
    job/workflow env, triggers, concurrency).
    """
    problems: list[str] = []
    doc = _load_yaml(ci_yml_text)
    wf = workflow_steps(ci_yml_text)
    if _meta(doc) != _meta(_load_yaml(CI_META_YAML)):
        problems.append("workflow/job-level keys differ (on/concurrency/runs-on/strategy/env/...)")
    table = [s for s in steps if not s.local_only]
    if len(wf) != len(table):
        problems.append(f"step count: ci.yml has {len(wf)}, local table has {len(table)}")
    table_by_key = {(s.job, s.name): s for s in table}
    for w in wf:
        s = table_by_key.get((str(w["job"]), str(w["name"])))
        if s is None:
            problems.append(f"ci.yml step not in table: [{w['job']}] {w['name']!r}")
            continue
        raw = dict(w["raw"])  # type: ignore[call-overload]
        run = raw.pop("run", None)
        if (run is None) != (s.ci_run is None) or (
            run is not None and norm(s.ci_run or "") != norm(str(run))
        ):
            problems.append(f"run text differs for {w['name']!r}")
        if raw != expected_step_dict(s):
            problems.append(f"step keys differ for {w['name']!r}: {raw} vs {expected_step_dict(s)}")
    wf_keys = {(str(w["job"]), str(w["name"])) for w in wf}
    problems.extend(f"table step not in ci.yml: {k!r}" for k in table_by_key if k not in wf_keys)
    # order within each job must match too
    for job in JOBS_ALL:
        wf_order = [str(w["name"]) for w in wf if w["job"] in (job, "*")]
        tb_order = [s.name for s in table if s.job in (job, "*")]
        if wf_order != tb_order:
            problems.append(f"step order differs for job {job}")
    return problems


# --------------------------------------------------------------------------------------------
# git / paths
# --------------------------------------------------------------------------------------------
def ci_root() -> Path:
    return Path(os.environ.get("FR_EN_CI_ROOT", str(DEFAULT_ROOT)))


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git with explicit argv in `repo`; a failure raises CiError carrying git's stderr."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and proc.returncode != 0:
        raise CiError(f"git {' '.join(args)} failed (rc {proc.returncode}): {proc.stderr.strip()}")
    return proc


def resolve_ref(repo: Path, ref: str) -> str:
    """Full 40-hex commit SHA for `ref` (sha, branch, tag, HEAD). Raises CiError if unknown."""
    proc = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    sha = proc.stdout.strip()
    if proc.returncode != 0 or not SHA_RE.match(sha):
        raise CiError(f"cannot resolve ref {ref!r} to a commit in {repo}")
    return sha


def job_key(job: str) -> str:
    return "py" + job.replace(".", "")


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def script_provenance(repo: Path, sha: str, script: Path | None = None) -> dict[str, object]:
    """sha256 of the running script (CRLF->LF) vs the blob committed at `sha`, so a provenance
    comparison is automatic. `matches_committed_blob` is None when `sha` has no such file."""
    script = script or Path(__file__).resolve()
    rel = script.resolve().relative_to(repo.resolve()).as_posix()
    local = hashlib.sha256(_lf(script.read_bytes())).hexdigest()
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{sha}:{rel}"], capture_output=True, check=False
    )
    blob = hashlib.sha256(_lf(proc.stdout)).hexdigest() if proc.returncode == 0 else None
    return {
        "path": rel,
        "running_sha256_lf": local,
        "committed_blob_sha256_lf": blob,
        "matches_committed_blob": None if blob is None else blob == local,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_cache_key(checkout: Path, job: str, interpreter_version: str) -> str:
    """Hash of the files that define a job's environment + the interpreter version."""
    files = ["uv.lock", "pyproject.toml"]
    if job == "3.13":
        files = ["requirements-colab.txt", "pyproject.toml", "colab/make_torch_constraints.py"]
    h = hashlib.sha256()
    h.update(f"{job}|{interpreter_version}|{PIP_PIN}|{TORCH_CPU_INDEX}".encode())
    for rel in files:
        h.update(rel.encode() + b"\0" + (checkout / rel).read_bytes() + b"\0")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Cleanup: check phase and delete phase are separate functions, called separately.
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class CleanupPlan:
    sha_dir: Path
    worktrees: list[Path]
    problems: list[str]
    links: list[Path] = dataclasses.field(default_factory=list)


REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT (junctions, symlinks, mount points, ...)


def _is_link(p: Path | str) -> bool:
    """True for symlinks AND Windows junctions/other reparse points (lstat, never follows)."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & REPARSE_POINT)


def find_links(root: Path) -> list[Path]:
    """Every symlink/junction/reparse point under `root` (root itself included), WITHOUT
    descending into any of them."""
    found: list[Path] = []
    if _is_link(root):
        return [root]
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError:
            continue
        for e in entries:
            if _is_link(e.path):
                found.append(Path(e.path))
            elif e.is_dir(follow_symlinks=False):
                stack.append(Path(e.path))
    return found


def _remove_link(link: Path) -> None:
    """Remove a link itself (never its target): unlink, else rmdir (directory junction)."""
    try:
        os.unlink(link)
    except (IsADirectoryError, PermissionError, OSError):
        os.rmdir(link)


def plan_cleanup(repo: Path, work_root: Path, sha: str, worktrees: Sequence[Path]) -> CleanupPlan:
    """CHECK phase (deletes nothing): is everything we are about to remove ours and in ci_work?

    A path is removable only if it resolves strictly inside work_root/<sha>, is not itself a link,
    (for worktrees) is a registered worktree of `repo`, and every link found inside the tree
    points inside work_root (links are removed first, never followed; one that points outside
    ci_work makes the plan refuse).
    """
    problems: list[str] = []
    sha_dir = work_root / sha
    root_r = work_root.resolve()
    links: list[Path] = []
    if not SHA_RE.match(sha):
        problems.append(f"not a full sha: {sha!r}")
    if sha_dir.exists():
        sd = sha_dir.resolve()
        if sd.parent != root_r:
            problems.append(f"{sd} is not directly under {root_r}")
        if _is_link(sha_dir):
            problems.append(f"{sha_dir} is a link")
        else:
            links = find_links(sha_dir)
            for link in links:
                target = link.resolve()
                if root_r not in target.parents:
                    problems.append(f"link {link} points outside ci_work: {target}")
    registered = {
        Path(line.split(" ", 1)[1]).resolve()
        for line in git(repo, "worktree", "list", "--porcelain").stdout.splitlines()
        if line.startswith("worktree ")
    }
    for wt in worktrees:
        if not wt.exists():
            continue
        if _is_link(wt):
            problems.append(f"{wt} is a link")
        elif wt.resolve().parent != sha_dir.resolve():
            problems.append(f"{wt} is not directly under {sha_dir}")
        elif wt.resolve() not in registered:
            problems.append(f"{wt} is not a registered worktree of {repo}")
    return CleanupPlan(sha_dir, [w for w in worktrees if w.exists()], problems, links)


def execute_cleanup(repo: Path, plan: CleanupPlan) -> None:
    """DELETE phase: only ever called with a plan whose check phase found no problems.

    Order: unlink every link (never following it), re-scan and require zero links, then
    `git worktree remove` (plain first; --force only after the re-scan found no links, because
    git's --force follows junctions on Windows), then remove the sha dir.
    """
    if plan.problems:
        raise CiError("refusing to clean up: " + "; ".join(plan.problems))
    if plan.sha_dir.exists():
        for link in find_links(plan.sha_dir):
            _remove_link(link)
        left = find_links(plan.sha_dir)
        if left:
            raise CiError(f"refusing to clean up: links remain after unlink: {left}")
    for wt in plan.worktrees:
        if find_links(wt):  # a link appeared after the scan: never hand it to git
            raise CiError(f"refusing to clean up: new link inside {wt}")
        if git(repo, "worktree", "remove", str(wt), check=False).returncode != 0:
            git(repo, "worktree", "remove", "--force", str(wt))  # links are gone (checked above)
    if plan.sha_dir.exists():
        if find_links(plan.sha_dir):
            raise CiError(f"refusing to rmtree {plan.sha_dir}: links present")
        shutil.rmtree(plan.sha_dir)


def stale_worktrees(repo: Path, work_root: Path) -> list[dict[str, object]]:
    """Registered worktrees of `repo` located directly in work_root/<sha>/, with sha and age."""
    root_r = work_root.resolve()
    rows: list[dict[str, object]] = []
    for line in git(repo, "worktree", "list", "--porcelain").stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        path = Path(line.split(" ", 1)[1])
        try:
            rp = path.resolve()
        except OSError:
            continue
        if rp.parent.parent == root_r and SHA_RE.match(rp.parent.name):
            age_h = (time.time() - rp.stat().st_mtime) / 3600 if rp.exists() else None
            rows.append(
                {
                    "path": path,
                    "sha": rp.parent.name,
                    "exists": rp.exists(),
                    "age_hours": None if age_h is None else round(age_h, 1),
                }
            )
    return rows


# --------------------------------------------------------------------------------------------
# Per-env lock: two concurrent runs must not install into / run from the same cached env.
# --------------------------------------------------------------------------------------------
LOCK_STALE_S = 6 * 3600


def pid_alive(pid: int) -> bool:
    """Is a process with this PID running? (Windows: OpenProcess; POSIX: signal 0.)"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def env_lock(lock_path: Path, wait_s: float, poll_s: float = 1.0):
    """Exclusive lockfile (PID + timestamp). Waits up to `wait_s`, then raises CiError.
    Yields the seconds spent waiting.

    A lock whose PID is dead, or older than LOCK_STALE_S, is taken over (noted on stderr).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    deadline = t_start + wait_s
    announced = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = ""
            try:
                holder = lock_path.read_text(encoding="utf-8").strip()
                pid_s, ts_s = holder.split()[:2]
                stale = (not pid_alive(int(pid_s))) or time.time() - float(ts_s) > LOCK_STALE_S
            except (OSError, ValueError):
                stale = False
            if stale:
                print(f"taking over stale lock {lock_path} ({holder})", file=sys.stderr)
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                continue
            if time.monotonic() >= deadline:
                raise CiError(
                    f"env lock {lock_path} is held by (pid ts) {holder!r}; gave up"
                ) from None
            if not announced:
                print(f"waiting for lock {lock_path} held by (pid ts) {holder!r}", flush=True)
                announced = True
            time.sleep(poll_s)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {time.time()}\n")
        break
    waited = round(time.monotonic() - t_start, 2)
    if announced:
        print(f"acquired lock {lock_path} after {waited}s", flush=True)
    try:
        yield waited  # seconds spent waiting (0.0 when free)
    finally:
        with contextlib.suppress(OSError):
            lock_path.unlink()


# --------------------------------------------------------------------------------------------
# Step execution
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Ctx:
    """Everything a step handler needs for one job."""

    job: str
    sha: str
    repo: Path
    root: Path
    checkout: Path
    runner_temp: Path
    env_dir: Path | None
    log: object  # text file handle, or None in dry mode
    uv: str
    dry: bool = False
    echo: bool = True
    lock_wait: float = 3600.0
    python: str | None = None  # interpreter used to create the 3.13 venv
    py_version: str = ""
    tail: deque[str] = dataclasses.field(default_factory=lambda: deque(maxlen=2000))
    extra: dict[str, object] = dataclasses.field(default_factory=dict)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    @property
    def venv(self) -> Path:
        assert self.env_dir is not None
        return self.env_dir / "venv"

    @property
    def venv_python(self) -> Path:
        # explicit translation of "$RUNNER_TEMP/venv/bin/python"
        return self.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def say(self, line: str) -> None:
        """Write a runner-originated line to the log (and console)."""
        self._emit(line)

    def _emit(self, line: str) -> None:
        self.tail.append(line)
        with self.lock:
            if self.log is not None:
                self.log.write(line + "\n")
                self.log.flush()
            if self.echo:
                with contextlib.suppress(Exception):
                    sys.stdout.write(f"[{self.job}] {line}\n")
                    sys.stdout.flush()

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
        env["RUNNER_TEMP"] = str(self.runner_temp)
        if self.env_dir is not None and self.job == "3.12":
            env["UV_PROJECT_ENVIRONMENT"] = str(self.env_dir / "venv")
        env.update(extra)
        return env

    def run(self, argv: Sequence[str], **env_extra: str) -> tuple[int, list[str]]:
        """Run argv in the checkout, streaming merged output to the log. Returns (rc, lines)."""
        self.say("$ " + " ".join(str(a) for a in argv))
        if self.dry:
            return 0, []
        lines: list[str] = []
        proc = subprocess.Popen(
            [str(a) for a in argv],
            cwd=self.checkout,
            env=self.env(**env_extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        timer = threading.Timer(STEP_TIMEOUT_S, lambda: _kill_tree(proc))
        timer.start()
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                lines.append(line)
                self._emit(line)
            rc = proc.wait()
        finally:
            timer.cancel()
        if rc != 0:
            self.say(f"(exit code {rc})")
        return rc, lines


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill our own timed-out child and its descendants."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


@dataclasses.dataclass
class StepResult:
    name: str
    status: str  # pass | fail | not_run
    duration_s: float = 0.0
    returncode: int | None = None
    detail: dict[str, object] = dataclasses.field(default_factory=dict)


Handler = Callable[[Ctx], tuple[int, dict[str, object]]]

_COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)")


def parse_pytest_summary(lines: Sequence[str]) -> dict[str, object]:
    """Counts from pytest's final line plus the `-rs` SKIPPED reasons.

    Works for both `-q` ("12 passed, 1 skipped in 3.2s") and `-v` ("==== 12 passed ... ====").
    """
    counts: dict[str, int] = {}
    for line in reversed(lines):
        if re.search(r" in [0-9.]+s", line) and _COUNT_RE.search(line):
            for n, kind in _COUNT_RE.findall(line):
                counts["errors" if kind.startswith("error") else kind] = int(n)
            break
    skips = [ln.strip() for ln in lines if ln.startswith("SKIPPED [")]
    return {"counts": counts, "skip_reasons": skips}


def _pytest_step(ctx: Ctx, argv: Sequence[str]) -> tuple[int, dict[str, object]]:
    rc, lines = ctx.run(argv, PYTEST_ADDOPTS="-rs")
    return rc, parse_pytest_summary(lines)


def _tail_step(ctx: Ctx, argv: Sequence[str]) -> tuple[int, dict[str, object]]:
    rc, lines = ctx.run(argv)
    return rc, {"tail": lines[-25:]}


def h_checkout(ctx: Ctx) -> tuple[int, dict[str, object]]:
    """Worktree is created by the orchestrator (before env hashing); verify it is the SHA."""
    if ctx.dry:
        ctx.say(f"(dry) git worktree add --detach {ctx.checkout} {ctx.sha}")
        return 0, {}
    head = git(ctx.checkout, "rev-parse", "HEAD").stdout.strip()
    ctx.say(f"checkout {ctx.checkout} at {head}")
    return (0 if head == ctx.sha else 1), {"head": head}


def h_install_uv(ctx: Ctx) -> tuple[int, dict[str, object]]:
    rc, lines = ctx.run([ctx.uv, "--version"])
    return rc, {"uv": lines[0] if lines else ""}


def h_uv_python_312(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _tail_step(ctx, [ctx.uv, "python", "install", "3.12"])


def h_uv_lock_check(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _tail_step(ctx, [ctx.uv, "lock", "--check"])


def h_uv_sync(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _tail_step(ctx, [ctx.uv, "sync", "--frozen", "--python", "3.12"])


def h_ruff_check(ctx: Ctx) -> tuple[int, dict[str, object]]:
    rc, lines = ctx.run([ctx.uv, "run", "ruff", "check", "."])
    return rc, {"tail": lines[-15:]}


def h_ruff_format(ctx: Ctx) -> tuple[int, dict[str, object]]:
    rc, lines = ctx.run([ctx.uv, "run", "ruff", "format", "--check", "."])
    return rc, {"tail": lines[-15:]}


def h_pytest_312(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _pytest_step(ctx, [ctx.uv, "run", "pytest", "-q"])


def h_scorer_encoding(ctx: Ctx) -> tuple[int, dict[str, object]]:
    """`pytest -v -rs | tee scorer_encoding.log`, then fail if any line contains ' SKIPPED'.

    Like ci.yml, a checkout without official/score.py (the public repo) passes without running.
    """
    if not ctx.dry and not (ctx.checkout / "official" / "score.py").is_file():
        ctx.say("official/score.py absent (public checkout): scorer-encoding step not applicable")
        return 0, {"applicable": False}
    if not ctx.dry:
        ctx.runner_temp.mkdir(parents=True, exist_ok=True)
    log_path = ctx.runner_temp / "scorer_encoding.log"
    argv = [
        ctx.uv,
        "run",
        "pytest",
        "tests/test_scorer_encoding.py",
        "tests/test_official_scorer.py",
        "-v",
        "-rs",
    ]
    rc, lines = ctx.run(argv)
    if not ctx.dry:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    detail = parse_pytest_summary(lines)
    detail["passed_lines"] = sum(1 for ln in lines if " PASSED" in ln)
    if rc == 0 and any(" SKIPPED" in ln for ln in lines):
        ctx.say("scorer-encoding tests were SKIPPED; they must run")
        rc = 1
    return rc, detail


def _interpreter_version(argv0: str) -> str:
    out = subprocess.run([argv0, "--version"], capture_output=True, text=True, check=False)
    return (out.stdout or out.stderr).strip()


def _freeze_hash(ctx: Ctx) -> tuple[int, str, int]:
    """(rc, sha256 of the sorted frozen requirement lines, package count).

    3.13 uses `pip list --format=freeze`, not `pip freeze`: freeze renders editable installs
    relative to the cwd and crashed with "path is on mount 'd:', start on mount 'C:'" when the
    interpreter's cwd/temp was on another drive; list --format=freeze prints name==version only.
    """
    if ctx.job == "3.12":
        argv = [ctx.uv, "pip", "freeze", "--python", str(ctx.env_dir / "venv" / _py_rel())]
    else:
        argv = [
            str(ctx.venv_python),
            "-m",
            "pip",
            "list",
            "--format=freeze",
            "--disable-pip-version-check",
        ]
    rc, lines = ctx.run(argv)
    lines = [ln for ln in lines if ln.strip() and not ln.startswith(("Traceback", " "))]
    return rc, hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest(), len(lines)


def _py_rel() -> str:
    return "Scripts/python.exe" if os.name == "nt" else "bin/python"


_VERIFY_REQS = r"""
import re, sys
from importlib import metadata
from packaging.markers import Marker
bad = []
for raw in open("requirements-colab.txt", encoding="utf-8"):
    line = raw.split("#", 1)[0].strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)\s*(?:;\s*(.*))?$", line)
    if not m:
        continue
    name, ver, marker = m.groups()
    if marker and not Marker(marker).evaluate():
        continue
    try:
        got = metadata.version(name)
    except metadata.PackageNotFoundError:
        bad.append(f"{name}: not installed (want {ver})")
        continue
    if got != ver:
        bad.append(f"{name}: installed {got}, want {ver}")
print("requirements-colab pin mismatches:", bad if bad else "none")
sys.exit(1 if bad else 0)
"""

_NMT_WHERE = "import nmt, pathlib; print(pathlib.Path(nmt.__file__).resolve())"


def _verify(ctx: Ctx, py: list[str]) -> tuple[int, dict[str, object]]:
    detail: dict[str, object] = {}
    rc, lines = ctx.run([*py, "-c", _NMT_WHERE])
    if not ctx.dry:
        where = Path(lines[-1]).resolve() if lines else Path("?")
        ok = ctx.checkout.resolve() in where.parents
        ctx.say(f"nmt imported from {where}; under the checkout under test: {ok}")
        detail["nmt_file"] = str(where)
        if rc == 0 and not ok:
            rc = 1
    if rc == 0 and ctx.job == "3.13":
        rc, _ = ctx.run([*py, "-c", _VERIFY_REQS])
    frc, digest, n = _freeze_hash(ctx)
    detail["freeze_sha256"] = digest
    detail["freeze_packages"] = n
    if not ctx.dry and (frc != 0 or n == 0):  # a failed/empty freeze must never read as PASS
        ctx.say(f"environment freeze failed (rc {frc}, {n} packages)")
        detail["freeze_rc"] = frc
        rc = rc or 1
    if not ctx.dry:
        st = git(ctx.checkout, "status", "--porcelain", "--untracked-files=no").stdout.strip()
        detail["tracked_files_modified_by_ci"] = st.splitlines()
    return rc, detail


def h_local_verify_312(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _verify(ctx, [ctx.uv, "run", "python"])


def h_local_verify_313(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _verify(ctx, [str(ctx.venv_python)])


def h_setup_python_313(ctx: Ctx) -> tuple[int, dict[str, object]]:
    """actions/setup-python 3.13 -> a uv-managed 3.13 interpreter (installed only if missing)."""
    rc, lines = ctx.run([ctx.uv, "python", "find", "3.13"])
    if ctx.dry:
        return 0, {}
    if rc != 0 or not lines:
        ctx.say("no 3.13 interpreter found; `uv python install 3.13` (uv-managed download)")
        rc, _ = ctx.run([ctx.uv, "python", "install", "3.13"])
        if rc != 0:
            return rc, {}
        rc, lines = ctx.run([ctx.uv, "python", "find", "3.13"])
        if rc != 0 or not lines:
            return 1, {}
    ctx.python = lines[-1].strip()
    ctx.py_version = _interpreter_version(ctx.python)
    ctx.say(f"python 3.13 -> {ctx.python} ({ctx.py_version})")
    return 0, {"python": ctx.python, "version": ctx.py_version}


def torch_version_from_pyproject(text: str) -> str:
    """`sed -n 's/^ *"torch==\\([0-9.]*\\)",$/\\1/p' pyproject.toml` (first match; '' if none)."""
    m = re.search(r'^ *"torch==([0-9.]*)",$', text, flags=re.MULTILINE)
    return m.group(1) if m else ""


def h_install_313(ctx: Ctx) -> tuple[int, dict[str, object]]:
    """Explicit translation of the ci.yml install block (one argv per shell command)."""
    pip = [str(ctx.venv_python), "-m", "pip", "install", "--disable-pip-version-check"]
    py = str(ctx.venv_python)
    constraints = ctx.runner_temp / "torch_constraints.txt"
    if not ctx.dry:
        ctx.runner_temp.mkdir(parents=True, exist_ok=True)
        ctx.venv.parent.mkdir(parents=True, exist_ok=True)
    assert ctx.python is not None or ctx.dry
    steps: list[Sequence[str]] = [[ctx.python or "python", "-m", "venv", str(ctx.venv)]]
    steps.append([*pip, f"pip=={PIP_PIN}"])
    for argv in steps:
        rc, _ = ctx.run(argv)
        if rc != 0:
            return rc, {}
    pyproject = (
        (ctx.checkout / "pyproject.toml").read_text(encoding="utf-8")
        if not ctx.dry
        else (git(ctx.repo, "show", f"{ctx.sha}:pyproject.toml").stdout)
    )
    torch_version = torch_version_from_pyproject(pyproject)
    ctx.say(f"TORCH_VERSION={torch_version!r}")
    if not torch_version:  # `test -n "$TORCH_VERSION"`
        ctx.say("TORCH_VERSION is empty")
        return 1, {}
    more: list[Sequence[str]] = [
        [*pip, f"torch=={torch_version}", "--index-url", TORCH_CPU_INDEX],
        [py, "colab/make_torch_constraints.py", str(constraints)],
        [*pip, "-r", "requirements-colab.txt", "-c", str(constraints)],
        [*pip, "-e", ".", "--no-deps"],
        [*pip, "pytest==9.1.1", "nbclient==0.11.0", "nbformat==5.11.1", "ipykernel==7.4.0"],
        [py, "-c", "import torch; print('torch', torch.__version__)"],
    ]
    for argv in more:
        rc, _ = ctx.run(argv)
        if rc != 0:
            return rc, {}
    # grep -qx "torch==<installed>" constraints  (whole-line match)
    rc, lines = ctx.run([py, "-c", "import torch; print(torch.__version__)"])
    if rc != 0:
        return rc, {}
    if not ctx.dry:
        want = f"torch=={lines[-1].strip()}"
        got = constraints.read_text(encoding="utf-8").splitlines()
        ctx.say(f"constraints file lines: {got}; need exact line {want!r}")
        if want not in got:
            return 1, {"torch": lines[-1].strip()}
    rc, lines = ctx.run([py, "-m", "pip", "check"])
    return rc, {"pip_check": lines[-3:]}


def h_pytest_313(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _pytest_step(ctx, [str(ctx.venv_python), "-m", "pytest", "-q"])


def _nb_argv(ctx: Ctx, *sets: str) -> list[str]:
    argv = [str(ctx.venv_python), "colab/execute_notebook.py", "colab/train.ipynb", "1800"]
    for s in sets:
        argv += ["--set", s]  # bash: --set CONFIG='"smoke"' passes CONFIG="smoke"
    return argv


def h_nb_smoke(ctx: Ctx) -> tuple[int, dict[str, object]]:
    rc, _ = ctx.run([str(ctx.venv_python), "colab/make_synthetic_shards.py", "."])
    if rc != 0:
        return rc, {}
    return _tail_step(
        ctx, _nb_argv(ctx, 'CONFIG="smoke"', "PLANNED_STEPS=None", "RESUME_TEST=False")
    )


def h_nb_ablations(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _tail_step(ctx, _nb_argv(ctx, 'CONFIG="ablations_l4"', "DRY_RUN=True"))


def h_nb_eval_all(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _tail_step(ctx, _nb_argv(ctx, 'CONFIG="eval_l4"', "DRY_RUN=True", 'RUN="all"'))


def check_needles(lines: Sequence[str], needles: Sequence[str]) -> list[str]:
    """The needles NOT present (as substrings) in the captured output: Python for ci.yml's
    `grep -qF` loop."""
    text = "\n".join(lines)
    return [n for n in needles if n not in text]


def h_nb_final_all(ctx: Ctx) -> tuple[int, dict[str, object]]:
    """ci.yml: run the notebook, `tee` the log, then `grep -qF` every COMET needle. Here the output
    is captured by ctx.run and the needles are asserted with Python (no bash/tee/grep)."""
    rc, lines = ctx.run(
        _nb_argv(ctx, 'CONFIG="eval_l4"', "DRY_RUN=True", 'RUN="final_all"'),
    )
    if rc != 0:
        return rc, {"tail": lines[-25:]}
    missing = check_needles(lines, FINAL_ALL_DRY_RUN_NEEDLES)
    for n in missing:
        ctx.say(f"final_all dry run lacks: {n}")
    return (1 if missing else 0), {"tail": lines[-25:], "missing_needles": missing}


def h_nb_extend(ctx: Ctx) -> tuple[int, dict[str, object]]:
    return _tail_step(ctx, _nb_argv(ctx, 'CONFIG="extend_l4"', "DRY_RUN=True"))


HANDLERS: dict[str, Handler] = {
    "checkout": h_checkout,
    "install_uv": h_install_uv,
    "uv_python_312": h_uv_python_312,
    "uv_lock_check": h_uv_lock_check,
    "uv_sync": h_uv_sync,
    "ruff_check": h_ruff_check,
    "ruff_format": h_ruff_format,
    "pytest_312": h_pytest_312,
    "scorer_encoding": h_scorer_encoding,
    "local_verify_312": h_local_verify_312,
    "setup_python_313": h_setup_python_313,
    "install_313": h_install_313,
    "pytest_313": h_pytest_313,
    "nb_smoke": h_nb_smoke,
    "nb_ablations": h_nb_ablations,
    "nb_eval_all": h_nb_eval_all,
    "nb_final_all": h_nb_final_all,
    "nb_extend": h_nb_extend,
    "local_verify_313": h_local_verify_313,
}


def job_steps(job: str) -> list[Step]:
    return [s for s in STEPS if s.job in (job, "*")]


def run_job(ctx: Ctx) -> dict[str, object]:
    """Run one job under an exclusive lock on its cached env (see env_lock)."""
    if ctx.dry or ctx.env_dir is None:
        return _run_steps(ctx)
    with env_lock(ctx.root / "ci_envs" / f"{ctx.env_dir.name}.lock", ctx.lock_wait) as waited:
        result = _run_steps(ctx)
    result["lock_wait_seconds"] = waited
    return result


def _run_steps(ctx: Ctx) -> dict[str, object]:
    """Run one job's steps in order; stop at the first failing step."""
    results: list[StepResult] = []
    failed: str | None = None
    for step in job_steps(ctx.job):
        if failed is not None:
            results.append(StepResult(step.name, "not_run"))
            continue
        ctx.say(f"::: step: {step.name}")
        t0 = time.monotonic()
        try:
            rc, detail = HANDLERS[step.handler](ctx)
        except Exception as exc:  # a crashing handler is a failing step, never a pass
            ctx.say(f"runner exception: {type(exc).__name__}: {exc}")
            rc, detail = 1, {"exception": f"{type(exc).__name__}: {exc}"}
        dur = round(time.monotonic() - t0, 2)
        status = "pass" if rc == 0 else "fail"
        label = "DRY (not run)" if ctx.dry and rc == 0 else status.upper()
        ctx.say(f"::: {label} ({dur}s): {step.name}")
        results.append(StepResult(step.name, status, dur, rc, detail))
        if rc != 0:
            failed = step.name
    return {
        "job": ctx.job,
        "status": "FAIL" if failed else ("DRY" if ctx.dry else "PASS"),
        "first_failing_step": failed,
        "python": ctx.py_version,
        "steps": [dataclasses.asdict(r) for r in results],
    }


def prepare_job(
    repo: Path, root: Path, sha: str, job: str, uv: str, *, dry: bool, echo: bool
) -> Ctx:
    """Create the clean checkout + env dirs and open the log for one job."""
    work = root / "ci_work" / sha
    checkout = work / job_key(job)
    logs = root / "ci_logs"
    ver = ""
    if job == "3.12":
        ver = _interpreter_version_uv(uv, "3.12")
    else:
        ver = _interpreter_version_uv(uv, "3.13")
    if not dry:
        logs.mkdir(parents=True, exist_ok=True)
        if checkout.exists():  # leftover from a crashed/--keep run: ours, so recycle via the checks
            plan = plan_cleanup(repo, root / "ci_work", sha, [checkout])
            execute_cleanup(repo, plan)
        work.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", "--detach", str(checkout), sha)
        key = env_cache_key(checkout, job, ver)
    else:
        key = "dry"
    env_dir = root / "ci_envs" / f"{job_key(job)}-{key}"
    if not dry:
        env_dir.mkdir(parents=True, exist_ok=True)
    log = None if dry else open(logs / f"{sha}__{job}.log", "w", encoding="utf-8")  # noqa: SIM115
    ctx = Ctx(
        job=job,
        sha=sha,
        repo=repo,
        root=root,
        checkout=checkout,
        runner_temp=work / f"_runner_temp_{job_key(job)}",
        env_dir=env_dir,
        log=log,
        uv=uv,
        dry=dry,
        echo=echo,
        py_version=ver,
    )
    return ctx


def _interpreter_version_uv(uv: str, spec: str) -> str:
    """Version string of the interpreter uv would pick for `spec` ('' if none installed yet)."""
    proc = subprocess.run([uv, "python", "find", spec], capture_output=True, text=True, check=False)
    path = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 and proc.stdout else ""
    return _interpreter_version(path) if path else f"uv-managed {spec} (not installed yet)"


def environment_info(uv: str) -> dict[str, object]:
    uv_v = subprocess.run([uv, "--version"], capture_output=True, text=True, check=False)
    git_v = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
    return {
        "platform": platform.platform(),
        "runner_python": sys.version.split()[0],
        "uv": uv_v.stdout.strip(),
        "git": git_v.stdout.strip(),
        "cpu_count": os.cpu_count(),
    }


def format_steps(job_filter: Sequence[str]) -> str:
    rows = []
    for job in job_filter:
        rows.append(f"== job {job} ==")
        for i, s in enumerate(job_steps(job), 1):
            tag = "local" if s.local_only else ("uses" if s.ci_uses else "run")
            rows.append(f"{i:2d}. [{tag:5s}] {s.name}")
    return "\n".join(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scripts.local_ci", description=__doc__)
    p.add_argument("--ref", default="HEAD", help="sha, branch or tag to test (default HEAD)")
    p.add_argument("--jobs", default=",".join(JOBS_ALL), help="comma list of: 3.12,3.13")
    p.add_argument("--keep", action="store_true", help="keep the ci_work checkouts afterwards")
    p.add_argument("--parallel", action="store_true", help="run the jobs concurrently")
    p.add_argument("--list-steps", action="store_true", help="print the step table and exit")
    p.add_argument("--dry", action="store_true", help="resolve and print every command only")
    p.add_argument("--allow-ci-drift", action="store_true", help="warn (not fail) on ci.yml drift")
    p.add_argument("--quiet", action="store_true", help="do not echo step output to the console")
    p.add_argument(
        "--prune",
        action="store_true",
        help="list registered worktrees under ci_work (sha, age); with --yes remove them safely",
    )
    p.add_argument("--yes", action="store_true", help="with --prune: actually remove the listed")
    p.add_argument(
        "--lock-wait",
        type=float,
        default=3600.0,
        help="seconds to wait for another run's env/sha lock before failing (0 = fail at once)",
    )
    args = p.parse_args(argv)
    args.job_list = [j.strip() for j in args.jobs.split(",") if j.strip()]
    bad = [j for j in args.job_list if j not in JOBS_ALL]
    if bad or not args.job_list:
        p.error(f"--jobs must be a comma list drawn from {JOBS_ALL}, got {args.jobs!r}")
    return args


def prune_main(repo: Path, root: Path, yes: bool) -> int:
    """List (and with --yes remove, via the safe path) registered worktrees under ci_work."""
    work_root = root / "ci_work"
    rows = stale_worktrees(repo, work_root)
    if not rows:
        print(f"no registered worktrees of {repo} under {work_root}")
        return 0
    for r in rows:
        print(f"{r['sha']}  {r['path']}  exists={r['exists']}  age_hours={r['age_hours']}")
    if not yes:
        print("listing only; nothing deleted. Re-run with --prune --yes to remove them.")
        return 0
    rc = 0
    for sha in sorted({str(r["sha"]) for r in rows}):
        paths = [Path(str(r["path"])) for r in rows if r["sha"] == sha and r["exists"]]
        plan = plan_cleanup(repo, work_root, sha, paths)
        print(f"cleanup check {sha}: {'OK' if not plan.problems else plan.problems}")
        if plan.problems:
            rc = 1
            continue
        try:
            execute_cleanup(repo, plan)
            print(f"removed {plan.sha_dir}")
        except CiError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            rc = 1
    return rc


def _cleanup(
    repo: Path, root: Path, sha: str, jobs: Sequence[str], keep: bool
) -> dict[str, object]:
    """CHECK phase, then (separately) DELETE phase; never raises (it runs in a `finally`)."""
    cleanup: dict[str, object] = {"performed": False, "problems": []}
    try:
        cands = [root / "ci_work" / sha / job_key(j) for j in jobs]
        plan = plan_cleanup(repo, root / "ci_work", sha, cands)
        cleanup["problems"] = list(plan.problems)
        print(f"cleanup check: {'OK' if not plan.problems else plan.problems}")
        if keep:
            print(f"--keep: leaving {plan.sha_dir}")
        elif not plan.problems:
            execute_cleanup(repo, plan)
            cleanup["performed"] = True
    except (CiError, OSError) as exc:
        cleanup["problems"] = [*cleanup["problems"], f"cleanup failed: {exc}"]  # type: ignore[misc]
        print(f"WARNING: cleanup failed: {exc}", file=sys.stderr)
    return cleanup


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    if args.list_steps:
        print(format_steps(args.job_list))
        return 0
    repo = REPO_ROOT
    root = ci_root()
    if args.prune:
        try:
            return prune_main(repo, root, args.yes)
        except CiError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    uv = shutil.which("uv")
    if uv is None:
        print("ERROR: uv is not on PATH", file=sys.stderr)
        return 2
    try:
        sha = resolve_ref(repo, args.ref)
    except CiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    started = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"local CI: ref={args.ref} sha={sha} jobs={args.job_list} dry={args.dry} root={root}")

    # ci.yml of the SHA under test must match the table (fail closed).
    ci_text = git(repo, "show", f"{sha}:.github/workflows/ci.yml", check=False)
    drift: list[str]
    if ci_text.returncode != 0:
        drift = ["no .github/workflows/ci.yml at that SHA"]
    else:
        try:
            drift = ci_drift(ci_text.stdout)
        except CiError as exc:
            drift = [str(exc)]
    if drift:
        print("ci.yml drift vs scripts/local_ci.py STEPS:\n  " + "\n  ".join(drift))
        if not args.allow_ci_drift:
            print("FAIL: ci.yml and local CI disagree; update STEPS (or --allow-ci-drift)")
            return 1

    contexts: list[Ctx] = []
    results: list[dict[str, object]] = []
    cleanup: dict[str, object] = {"performed": False, "problems": []}
    error: str | None = None
    sha_lock_wait = 0.0
    t_start = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            if not args.dry:  # one run per SHA at a time: they share ci_work/<sha>/
                sha_lock_wait = stack.enter_context(
                    env_lock(root / "ci_envs" / f"sha-{sha}.lock", args.lock_wait)
                )
            try:
                for job in args.job_list:
                    ctx = prepare_job(repo, root, sha, job, uv, dry=args.dry, echo=not args.quiet)
                    ctx.lock_wait = args.lock_wait
                    contexts.append(ctx)
                if args.parallel and len(contexts) > 1:
                    with concurrent.futures.ThreadPoolExecutor(len(contexts)) as ex:
                        results = list(ex.map(run_job, contexts))
                else:
                    results = [run_job(c) for c in contexts]
            finally:  # also on CiError / Ctrl-C: never leave registered worktrees behind
                for c in contexts:
                    if c.log is not None:
                        c.log.close()
                if not args.dry:
                    cleanup = _cleanup(repo, root, sha, args.job_list, args.keep)
    except (CiError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc)
    except KeyboardInterrupt:
        error = "interrupted"
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.dry:
        for r in results:
            print(f"planned job {r['job']}: {len(r['steps'])} steps")  # type: ignore[arg-type]
        print("DRY RUN (nothing executed)")
        return 3

    ok = all(r["status"] == "PASS" for r in results)
    summary: dict[str, object] = {
        "sha": sha,
        "ref": args.ref,
        "status": "PASS" if ok else "FAIL",
        "jobs": results,
        "ci_yml_drift": drift,
        "environment": environment_info(uv),
        "timestamp_utc": started,
        "duration_s": round(time.monotonic() - t_start, 1),
        "script_sha256": sha256_file(Path(__file__)),
        "script_provenance": script_provenance(repo, sha),
        "sha_lock_wait_seconds": sha_lock_wait,
        "exit_code": 0 if ok else 1,
        "parallel": bool(args.parallel),
        "cleanup": cleanup,
    }
    out = root / "ci_logs" / f"{sha}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out}")
    for r in results:
        failing = r["first_failing_step"]
        print(
            f"{r['status']} job {r['job']}"
            + (f" -- first failing step: {failing!r}" if failing else "")
        )
    print(f"{'PASS' if ok else 'FAIL'} {sha} ({summary['duration_s']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
