from __future__ import annotations

# Executes colab/train.ipynb top-to-bottom with nbclient (cwd = colab/, so the notebook's local
# mode finds the repo one level up) WITHOUT writing outputs back: the committed notebook stays
# output-free. Prints every code cell's stdout so the CI log shows what a user would see, and
# exits non-zero on the first failing cell. Colab-only cells skip themselves in local mode.
#
# The notebook's Parameters cell defaults to the real L4 main run (CONFIG="main"), so a CI/local
# smoke execution must say so explicitly: `--set NAME=<python literal>` rewrites that top-level
# assignment in memory before executing (an unknown NAME is an error), and the run is refused
# unless CONFIG ends up "smoke" (a CPU "main" run would never finish); --allow-non-smoke
# overrides that guard knowingly.
#
# Usage: python colab/execute_notebook.py [notebook-path] [timeout-seconds]
#            [--set NAME=LITERAL ...] [--allow-non-smoke]
#   e.g. --set CONFIG='"smoke"' --set PLANNED_STEPS=None --set RESUME_TEST=False
import ast
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _params_cell(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    return next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith("# --- Param"))


def apply_overrides(nb: nbformat.NotebookNode, overrides: dict[str, str]) -> None:
    """Rewrite top-level `NAME = ...` assignments in the Parameters cell (keeping any trailing
    comment). Raises ValueError for a non-literal value or a name with no such assignment.
    """
    cell = _params_cell(nb)
    for name, literal in overrides.items():
        try:
            ast.literal_eval(literal)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"--set {name}: {literal!r} is not a Python literal") from exc
        pattern = re.compile(rf"^{re.escape(name)} = [^#\n]*?(\s*#.*)?$", re.MULTILINE)
        new_source, count = pattern.subn(
            lambda m, n=name, v=literal: f"{n} = {v}" + (m.group(1) or ""), cell.source, count=1
        )
        if count != 1:
            raise ValueError(f"no top-level `{name} = ...` assignment in the Parameters cell")
        cell.source = new_source


def effective_config(nb: nbformat.NotebookNode) -> str:
    """The Parameters cell's CONFIG value (after any overrides)."""
    match = re.search(r'^CONFIG = "(\w+)"', _params_cell(nb).source, re.MULTILINE)
    if match is None:
        raise ValueError('Parameters cell has no `CONFIG = "<name>"` assignment')
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    allow_non_smoke = "--allow-non-smoke" in args
    args = [a for a in args if a != "--allow-non-smoke"]
    overrides: dict[str, str] = {}
    while "--set" in args:
        i = args.index("--set")
        name, sep, literal = args[i + 1].partition("=") if i + 1 < len(args) else ("", "", "")
        if not sep:
            print("usage: --set NAME=LITERAL", file=sys.stderr)
            return 2
        overrides[name] = literal
        del args[i : i + 2]
    path = Path(args[0]) if args else Path(__file__).with_name("train.ipynb")
    timeout = int(args[1]) if len(args) > 1 else 1800
    nb = nbformat.read(path, as_version=4)
    apply_overrides(nb, overrides)
    config = effective_config(nb)
    if config != "smoke" and not allow_non_smoke:
        print(
            f"refusing to execute with CONFIG={config!r} (a CPU run of it never finishes): pass "
            "--set CONFIG='\"smoke\"' (or --allow-non-smoke).",
            file=sys.stderr,
        )
        return 2
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
