from __future__ import annotations

# Writes a pip constraints file pinning torch to exactly the build that is ALREADY installed
# (full version incl. the local tag, e.g. `torch==2.9.0+cu126`). Shared by colab/train.ipynb and
# the CI `py313` job so the two cannot drift: the notebook never reinstalls Colab's preinstalled
# CUDA torch, it only forbids pip from replacing it while resolving requirements-colab.txt.
# Reads the version from package metadata (no `import torch`: instant, no CUDA init).
#
# Usage: python colab/make_torch_constraints.py <output-path>
import sys
from importlib import metadata
from pathlib import Path


def torch_constraint() -> str:
    """`torch==<installed version>` (local tag kept). Raises if torch is not installed."""
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "torch is not installed in this interpreter; on Colab it is preinstalled (check "
            "Runtime type), in CI install the CPU wheel first."
        ) from exc
    return f"torch=={version}"


def write_constraints(path: Path) -> str:
    """Write the constraint line to `path` (parents created); returns the line written."""
    line = torch_constraint()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")
    return line


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: make_torch_constraints.py <output-path>", file=sys.stderr)
        return 2
    print(write_constraints(Path(args[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
