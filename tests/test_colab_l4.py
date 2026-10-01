from __future__ import annotations

# Colab L4 main-run plumbing: the notebook's Parameters defaults, the single-GPU <= 24 GiB
# preflight rule + resolved precision, the W&B RUN-url resolution (the old "last wandb.ai URL
# wins" scan printed the project URL), and CI's explicit smoke override of the notebook defaults.
# Notebook cells are exec'd from the real notebook (not copies), like tests/test_colab_helpers.py.
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "colab" / "train.ipynb"
GIB = 1024**3
GPU_CELL, URL_CELL, PARAMS_CELL = "e1a7c3b5", "f4b8d2a6", "c5459372"


def _cell_source(cell_id: str) -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    return next(c.source for c in nb.cells if c.id == cell_id)


def _exec_cell(cell_id: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"Path": Path, "re": re}
    exec(compile(_cell_source(cell_id), f"<notebook cell {cell_id}>", "exec"), namespace)  # noqa: S102
    return namespace


def _load_execute_notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "colab_execute_notebook", REPO_ROOT / "colab" / "execute_notebook.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parameters_cell_defaults_are_the_l4_main_run() -> None:
    lines = _cell_source(PARAMS_CELL).splitlines()
    for expected in (
        'CONFIG = "main"',
        "PLANNED_STEPS = 24645",
        "RESUME_TEST = False",
        'GIT_REF = "v0.2.2-colab"',
    ):
        assert any(ln.startswith(expected) for ln in lines), expected
    assert 'ALLOWED_CONFIGS = ("smoke", "pilot", "main")' in lines  # pilot/smoke stay selectable


# --- preflight GPU rule -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("devices", "ok"),
    [
        ([("Tesla T4", int(14.56 * GIB))], True),  # T4: 15360 MiB
        ([("NVIDIA L4", int(22.49 * GIB))], True),  # L4 reports ~22.5 GiB
        ([("NVIDIA L4", 24_152_899_584)], True),  # > 24e9 bytes: why the limit is GiB, not GB
        ([("NVIDIA A100-SXM4-40GB", int(39.4 * GIB))], False),
        ([("NVIDIA A100-SXM4-80GB", int(79.2 * GIB))], False),
        ([("NVIDIA L4", int(22.49 * GIB))] * 2, False),  # two GPUs, each individually fine
        ([("X", 24 * GIB)], True),  # boundary: exactly 24 GiB passes
        ([("X", 24 * GIB + 1)], False),  # one byte over fails
        ([], False),  # no GPU
    ],
)
def test_gpu_constraint_is_one_gpu_of_at_most_24_gib(
    devices: list[tuple[str, int]], ok: bool
) -> None:
    check = _exec_cell(GPU_CELL)["check_gpu_constraint"]
    if ok:
        check(devices)
    else:
        with pytest.raises(RuntimeError, match="preflight"):
            check(devices)


@pytest.mark.parametrize(
    ("cuda", "bf16", "expected"),
    [(True, True, "bf16"), (True, False, "fp16"), (False, False, "fp32")],
)
def test_resolved_precision_matches_nmt_train_auto_rule(
    cuda: bool, bf16: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nmt.train import resolve_precision

    assert _exec_cell(GPU_CELL)["resolve_auto_precision"](cuda, bf16) == expected
    if cuda:  # cross-check against the real function with a faked bf16 capability
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
        assert resolve_precision("auto", torch.device("cuda")) == expected


def test_preflight_cell_logs_gpu_memory_and_precision_and_has_no_t4_assertion() -> None:
    src = _cell_source("d9a5b1e7")
    for key in ("preflight_gpu_count", "preflight_gpu_total_memory_gib", "preflight_precision"):
        assert f'"{key}"' in src
    assert "check_gpu_constraint(gpu_devices)" in src and "Tesla T4" not in src


# --- W&B run URL --------------------------------------------------------------------------------

PROJECT_URL = "https://wandb.ai/ent/proj"
RUN_URL = f"{PROJECT_URL}/runs/abc12345"
WANDB_TAIL = [
    "wandb: Syncing run main",
    f"wandb: View project at {PROJECT_URL}",
    f"wandb: View run main at: {RUN_URL}",
    "train: done",
    f"wandb: View run main at: {RUN_URL}\x1b[0m",
    f"wandb: View project at: {PROJECT_URL}",  # printed LAST: what the old scan returned
]


def test_run_url_comes_from_the_id_file_and_is_never_the_project_url(tmp_path: Path) -> None:
    resolve = _exec_cell(URL_CELL)["resolve_wandb_run_url"]
    (tmp_path / "wandb_run_id.txt").write_text("abc12345\n", encoding="utf-8")
    assert resolve(tmp_path, WANDB_TAIL, "ent", "proj") == RUN_URL
    # id file only (no wandb output): built from the notebook's entity/project
    assert resolve(tmp_path, [], "e2", "p2") == "https://wandb.ai/e2/p2/runs/abc12345"
    # entity/project wandb actually used (its "View run" line) win over the parameters
    assert resolve(tmp_path, WANDB_TAIL, "other", "other") == RUN_URL


def test_run_url_falls_back_to_the_view_run_line_without_an_id_file(tmp_path: Path) -> None:
    resolve = _exec_cell(URL_CELL)["resolve_wandb_run_url"]
    # the trailing ANSI escape must not leak into the URL; the later project URL must not win
    assert resolve(tmp_path, WANDB_TAIL, "ent", "proj") == RUN_URL
    assert resolve(tmp_path, [f"View project at {PROJECT_URL}"], "ent", "proj") is None
    assert resolve(tmp_path, [], "ent", "proj") is None


def test_train_cell_uses_the_resolver_not_a_last_url_scan() -> None:
    train = _cell_source("4e5fb4a0")
    assert "resolve_wandb_run_url(" in train and "wandb\\.ai/\\S+" not in train


# --- CI / local smoke override ---------------------------------------------------------------


def test_execute_notebook_overrides_defaults_and_refuses_a_non_smoke_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_execute_notebook()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert mod.effective_config(nb) == "main"  # the committed default is the real run
    mod.apply_overrides(nb, {"CONFIG": '"smoke"', "PLANNED_STEPS": "None", "RESUME_TEST": "False"})
    params = next(c.source for c in nb.cells if c.source.startswith("# --- Parameters"))
    assert 'CONFIG = "smoke"  # one of' in params  # trailing comment kept
    assert "PLANNED_STEPS = None" in params and "RESUME_TEST = False" in params
    assert mod.effective_config(nb) == "smoke"
    with pytest.raises(ValueError, match="no top-level"):
        mod.apply_overrides(nb, {"NOT_A_PARAM": "1"})
    with pytest.raises(ValueError, match="not a Python literal"):
        mod.apply_overrides(nb, {"CONFIG": "smoke"})
    assert mod.main([str(NOTEBOOK)]) == 2  # defaults (CONFIG=main) are refused on a CPU host
    assert "refusing to execute" in capsys.readouterr().err


def test_ci_sets_the_notebook_to_smoke_explicitly() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--set CONFIG='\"smoke\"' --set PLANNED_STEPS=None --set RESUME_TEST=False" in ci
