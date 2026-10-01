from __future__ import annotations

# Drift check for envs/cuda (the local-GPU uv project): every package it locks must be pinned to
# the identical version as the root uv.lock (the CPU/CI env), except torch's local version tag
# (+cu130 vs +cpu) -- so a run on the 3070 and a run in CI can only differ by the CUDA wheel.
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_ENV = REPO_ROOT / "envs" / "cuda"

# Packages that exist only because of the CUDA torch wheel (its nvidia-* runtime pins, cuda-*
# helpers, triton on Linux) plus the env's own virtual project.
_CUDA_ONLY = re.compile(r"^(nvidia-.*|cuda-.*|triton|fr-en-transformer-cuda)$")


def _locked_versions(lock_path: Path) -> dict[str, str]:
    """name -> version with torch's `+local` tag stripped (the root lock holds torch twice)."""
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    return {p["name"]: p["version"].split("+")[0] for p in lock["package"]}


def _direct_dependencies(pyproject_path: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"] + data.get("dependency-groups", {}).get("dev", [])
    return dict(d.split("==") for d in deps)


def test_shared_packages_have_identical_pinned_versions() -> None:
    main = _locked_versions(REPO_ROOT / "uv.lock")
    cuda = _locked_versions(CUDA_ENV / "uv.lock")
    mismatched = {
        name: (main[name], cuda[name])
        for name in main.keys() & cuda.keys()
        if main[name] != cuda[name]
    }
    assert not mismatched, f"envs/cuda/uv.lock drifted from uv.lock (main, cuda): {mismatched}"
    assert main["torch"] == cuda["torch"] == "2.14.0"


def test_cuda_env_neither_drops_nor_invents_packages() -> None:
    main = _locked_versions(REPO_ROOT / "uv.lock")
    cuda = _locked_versions(CUDA_ENV / "uv.lock")
    dropped = {n for n in main if n not in cuda and n != "fr-en-transformer"}
    extra = {n for n in cuda if n not in main and not _CUDA_ONLY.match(n)}
    assert not dropped, f"packages in uv.lock missing from envs/cuda/uv.lock: {sorted(dropped)}"
    assert not extra, f"unexpected non-CUDA packages only in envs/cuda/uv.lock: {sorted(extra)}"


def test_cuda_env_direct_pins_match_root_pyproject() -> None:
    root = _direct_dependencies(REPO_ROOT / "pyproject.toml")
    cuda = _direct_dependencies(CUDA_ENV / "pyproject.toml")
    assert root == cuda


def test_cuda_env_torch_comes_from_a_cuda_index_and_root_stays_cpu() -> None:
    cuda_lock = (CUDA_ENV / "uv.lock").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cu130" in cuda_lock
    assert "+cu130" in cuda_lock
    root_lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert "+cu" not in root_lock  # CI's lock must stay CPU-only
