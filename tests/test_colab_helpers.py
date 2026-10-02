from __future__ import annotations

# Tests for the Colab-on-Python-3.13 plumbing: the torch-constraints helper (shared by the
# notebook and CI), the W&B preflight-config launcher, the synthetic-shard builder CI uses for the
# notebook smoke run, the notebook's `run_step` failure helper (executed from the real notebook
# cell, not a copy), and static guards on requirements-colab.txt / the notebook defaults.
import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat
import pytest
import torch

from nmt.data.hub_data import verify_shards_against_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB = REPO_ROOT / "colab"
NOTEBOOK = COLAB / "train.ipynb"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"colab_{name}", COLAB / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_source(cell_id: str) -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    return next(c.source for c in nb.cells if c.id == cell_id)


@pytest.fixture(scope="module")
def run_step() -> Callable[..., str]:
    """The notebook's own `run_step`, exec'd from its helper cell."""
    namespace: dict[str, Any] = {"Path": Path}
    exec(compile(_cell_source("b7e3a9c1"), "<notebook helper cell>", "exec"), namespace)  # noqa: S102
    return namespace["run_step"]


# --- make_torch_constraints ---------------------------------------------------------------


def test_torch_constraint_keeps_the_full_local_version_tag(tmp_path: Path) -> None:
    mod = _load("make_torch_constraints")
    line = mod.write_constraints(tmp_path / "sub" / "c.txt")
    # Full version incl. the +cpu / +cuXXX tag: pip must see the exact installed build.
    assert line == f"torch=={torch.__version__}"
    assert (tmp_path / "sub" / "c.txt").read_text(encoding="utf-8") == line + "\n"


def test_torch_constraint_fails_loudly_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load("make_torch_constraints")

    def _missing(_name: str) -> str:
        raise mod.metadata.PackageNotFoundError("torch")

    monkeypatch.setattr(mod.metadata, "version", _missing)
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.torch_constraint()


def test_constraints_cli_requires_exactly_one_argument() -> None:
    mod = _load("make_torch_constraints")
    assert mod.main([]) == 2


# --- run_train: preflight YAML -> W&B config ------------------------------------------------


def test_preflight_yaml_is_loadable_by_wandb_and_coerces_str_subclasses(tmp_path: Path) -> None:
    from wandb.sdk.lib import config_util

    class _Version(str):  # stands in for torch.TorchVersion, which yaml.safe_dump rejects
        pass

    mod = _load("run_train")
    path = tmp_path / "preflight.yaml"
    values = {"preflight_torch": _Version("2.9.0+cu126"), "preflight_cudnn": 91002, "x": None}
    mod.write_wandb_config_yaml(path, values)
    assert config_util.dict_from_config_file(str(path)) == {
        "preflight_torch": "2.9.0+cu126",
        "preflight_cudnn": 91002,
        "x": None,
    }


def test_wandb_settings_route_merges_the_yaml_into_run_config(tmp_path: Path) -> None:
    # Subprocess: wandb.setup() is a process-global singleton; keep it out of the other tests.
    path = tmp_path / "preflight.yaml"
    _load("run_train").write_wandb_config_yaml(path, {"preflight_gpu": "Tesla T4"})
    code = (
        "import json, sys, wandb; "
        "s = wandb.setup(settings=wandb.Settings(config_paths=[sys.argv[1]])); "
        "print(json.dumps(s.config))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"preflight_gpu": "Tesla T4"}


# --- make_synthetic_shards ------------------------------------------------------------------


def test_synthetic_shards_pass_the_real_manifest_verifier_and_never_overwrite(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer").mkdir()
    shutil.copy(REPO_ROOT / "tokenizer" / "spm.model", tmp_path / "tokenizer" / "spm.model")
    mod = _load("make_synthetic_shards")
    shards = mod.build(tmp_path)
    assert shards == tmp_path / "data" / "shards"
    manifest = verify_shards_against_manifest(tmp_path)
    assert set(manifest["shard_files_sha256"]) == {f"{n}/shard_00000.npz" for n in mod.SPLITS}
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        mod.build(tmp_path)


# --- notebook run_step ----------------------------------------------------------------------


def test_run_step_success_returns_stdout(run_step: Callable[..., str]) -> None:
    assert run_step("t: echo", [sys.executable, "-c", "print('hi')"]) == "hi"


def test_run_step_failure_names_step_and_prints_last_80_lines(
    run_step: Callable[..., str], capsys: pytest.CaptureFixture[str]
) -> None:
    code = (
        "import sys; [print(f'line{i}') for i in range(100)]; "
        "print('boom', file=sys.stderr); sys.exit(3)"
    )
    with pytest.raises(RuntimeError, match=r"t: failing step failed \(exit 3\)"):
        run_step("t: failing step", [sys.executable, "-c", code])
    shown = capsys.readouterr().out
    assert "boom" in shown  # stderr is part of the combined tail
    assert "line99" in shown
    assert "line20" not in shown  # 101 lines total -> only the last 80 (line21+) are shown
    assert "line21" in shown


def test_run_step_scrubs_secrets_from_output_and_log(
    run_step: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghp_SECRETSECRET")
    code = "import sys; print('token=ghp_SECRETSECRET'); sys.exit(1)"
    log = tmp_path / "log.txt"
    with pytest.raises(RuntimeError) as err:
        run_step("t: leak", [sys.executable, "-c", code], stream=True, log_path=log)
    assert "ghp_SECRETSECRET" not in capsys.readouterr().out
    assert "ghp_SECRETSECRET" not in str(err.value)
    assert "ghp_SECRETSECRET" not in log.read_text(encoding="utf-8")
    assert "token=***" in log.read_text(encoding="utf-8")


def test_run_step_check_false_and_missing_executable_return_empty(
    run_step: Callable[..., str],
) -> None:
    assert run_step("t: nope", ["definitely-not-a-real-binary-xyz"], check=False) == ""
    with pytest.raises(RuntimeError, match=r"exit 127"):
        run_step("t: nope", ["definitely-not-a-real-binary-xyz"])


# --- static guards --------------------------------------------------------------------------


def test_requirements_colab_never_reinstalls_torch_or_the_project() -> None:
    lines = [
        ln.strip()
        for ln in (REPO_ROOT / "requirements-colab.txt").read_text(encoding="utf-8").splitlines()
    ]
    pins = [ln for ln in lines if ln and not ln.startswith("#")]
    assert not [ln for ln in pins if ln.lower().startswith("torch")]
    assert "-e ." not in pins


def _pins(path: Path) -> dict[str, str]:
    """name -> version for every `name==version` line (env markers stripped), torch excluded."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:1].isalpha() and "==" in line:
            name, version = line.split(";")[0].strip().split("==")
            pins[name] = version
    pins.pop("torch", None)
    return pins


def test_requirements_colab_pins_equal_requirements_txt_minus_torch() -> None:
    # requirements.txt is the uv export of uv.lock; the Colab file must not drift from it.
    assert _pins(REPO_ROOT / "requirements-colab.txt") == _pins(REPO_ROOT / "requirements.txt")


def test_notebook_is_output_free_tag_pinned_and_has_no_torch_install_logic() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    params = next(c.source for c in code if c.source.startswith("# --- Parameters"))
    assert 'GIT_REF = "v0.2.4-colab"' in params
    joined = "\n".join(c.source for c in code)
    assert "ALLOW_NON_T4" not in joined  # the GPU rule has no override (see test_colab_l4.py)
    assert "PINNED_TORCH_VERSION" not in joined
    assert "subprocess.run(" not in joined  # every subprocess goes through run_step


def test_torch_api_surface_used_by_cuda_only_code_paths_exists() -> None:
    # nmt.train touches these only on CUDA, so the CPU suite alone would not notice a torch that
    # lacks them. Backs MIN_TORCH = "2.4" in the notebook's PREFLIGHT cell.
    import inspect

    from torch.nn.attention import SDPBackend, sdpa_kernel

    assert sdpa_kernel is not None
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        assert hasattr(SDPBackend, name)
    assert hasattr(torch, "OutOfMemoryError") and hasattr(torch.amp, "GradScaler")
    for name in (
        "mem_get_info",
        "set_per_process_memory_fraction",
        "is_bf16_supported",
        "get_rng_state_all",
        "set_rng_state_all",
        "reset_peak_memory_stats",
        "max_memory_reserved",
    ):
        assert hasattr(torch.cuda, name), name
    assert "devices" in inspect.signature(torch.random.fork_rng).parameters


# --- kernel import rule + subprocess helpers --------------------------------------------------


def test_no_cell_after_install_imports_non_stdlib_except_torch_and_google_colab() -> None:
    # Colab's kernel already holds its preinstalled numpy/pandas/...; pip may upgrade them on disk
    # during the install cell. Any later in-kernel import of project code or a third-party
    # package could mix versions, so after the install cell only stdlib / torch / google.colab
    # may be imported in the kernel; everything else must run as a subprocess via run_step.
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    install_idx = next(i for i, c in enumerate(code) if c.id == "3cfbc1cb")
    offenders: list[str] = []
    for cell in code[install_idx:]:
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import in a notebook cell"
                names = [node.module or ""]
            else:
                continue
            for name in names:
                allowed = (
                    name.split(".")[0] in sys.stdlib_module_names
                    or name.split(".")[0] == "torch"
                    or name == "google.colab"
                    or name.startswith("google.colab.")
                )
                if not allowed:
                    offenders.append(f"cell {cell.id}: import {name}")
    assert not offenders, offenders


def test_notebook_has_no_in_kernel_importlib_escape_hatch() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "import_module(" not in joined and "__import__(" not in joined


def test_preflight_json_is_converted_to_wandb_yaml_by_the_launcher(tmp_path: Path) -> None:
    from wandb.sdk.lib import config_util

    values = {"preflight_torch": "2.9.0+cu126", "preflight_cudnn": None, "preflight_gpu": "x"}
    src = tmp_path / "preflight_config.json"
    src.write_text(json.dumps(values), encoding="utf-8")
    out = _load("run_train").prepare_preflight_yaml(src)
    assert out == tmp_path / "preflight_config.yaml"
    assert config_util.dict_from_config_file(str(out)) == values


class _FakeApi:
    """Stands in for wandb.Api: `_service_api.execute_graphql` yields scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.calls = 0
        self._responses = responses
        self._service_api = self

    def execute_graphql(self, _query: str, variables: dict[str, str]) -> Any:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def test_wandb_privacy_check_passes_only_for_private_and_retries_a_null_first_lookup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load("check_wandb_private")
    api = _FakeApi([{"project": None}, {"project": {"access": "PRIVATE"}}])
    assert mod.verify_project_is_private("e", "p", api=api, sleep=lambda _s: None) == "PRIVATE"
    assert api.calls == 2
    assert "W&B project e/p: access=PRIVATE" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("responses", "match"),
    [
        ([{"project": {"access": "PUBLIC"}}], "not PRIVATE"),
        ([{"project": None}], "does not exist"),
        ([RuntimeError("boom")], "Could not verify"),
    ],
)
def test_wandb_privacy_check_fails_closed(responses: list[Any], match: str) -> None:
    mod = _load("check_wandb_private")
    api = _FakeApi(responses)
    with pytest.raises(RuntimeError, match=match):
        mod.verify_project_is_private("e", "p", api=api, sleep=lambda _s: None)


def test_wandb_privacy_check_cli_exits_nonzero_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _load("check_wandb_private").main(["e", "p"]) == 1
    assert "FAILED" in capsys.readouterr().err
