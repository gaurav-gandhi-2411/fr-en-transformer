from __future__ import annotations

# Executes colab/train.ipynb top-to-bottom with nbclient (cwd = colab/, so the notebook's local
# mode finds the repo one level up) WITHOUT writing outputs back: the committed notebook stays
# output-free. Prints every code cell's stdout so the CI log shows what a user would see, and
# exits non-zero on the first failing cell. CONFIG is whatever the Parameters cell says
# ("smoke" by default: CPU, offline W&B); Colab-only cells skip themselves in local mode.
#
# Usage: python colab/execute_notebook.py [notebook-path] [timeout-seconds]
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else Path(__file__).with_name("train.ipynb")
    timeout = int(args[1]) if len(args) > 1 else 1800
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        _print_outputs(nb)
        print(f"NOTEBOOK FAILED: {exc}", file=sys.stderr)
        return 1
    _print_outputs(nb)
    print(f"NOTEBOOK OK: {sum(c.cell_type == 'code' for c in nb.cells)} code cells executed")
    return 0


def _print_outputs(nb: nbformat.NotebookNode) -> None:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        text = "".join(
            o.get("text", "") for o in cell.get("outputs", []) if o.output_type == "stream"
        )
        if text.strip():
            print(f"--- cell {i} ---\n{text.rstrip()}")


if __name__ == "__main__":
    raise SystemExit(main())
