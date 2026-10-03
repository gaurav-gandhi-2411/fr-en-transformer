from __future__ import annotations

# Verify the `file:start-end` line references in docs/WALKTHROUGH.md against the code. Two row
# forms are understood, and any other `path.py:a-b` token in the file is an error (an unchecked
# reference would silently rot):
#   symbol row:  `Name` | `nmt/x.py:10-20`
#       Name must be a def or class spanning exactly lines 10-20
#   anchor row:  `nmt/x.py:10-20` anchor `some text`
#       the text must occur on a line within 10-20
# A `pr23:` prefix on the path reads the file from a git ref (default origin/feat/mbr-ensemble)
# instead of the working tree, for code that is not on main yet; it is skipped with a warning
# when the ref does not exist.
#
# Usage: python -m scripts.check_doc_ranges [docs/WALKTHROUGH.md] [--pr23-ref REF]
import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = REPO_ROOT / "docs" / "WALKTHROUGH.md"
DEFAULT_PR23_REF = "origin/feat/mbr-ensemble"

_PATH = r"(?P<pr>pr23:)?(?P<file>[\w/]+\.py):(?P<a>\d+)-(?P<b>\d+)"
_SYMBOL_ROW = re.compile(rf"`(?P<sym>[A-Za-z_][\w.]*)` \| `{_PATH}`")
_ANCHOR_ROW = re.compile(rf"`{_PATH}` anchor `(?P<txt>[^`]+)`")
_ANY_REF = re.compile(r"`(?:pr23:)?[\w/]+\.py:\d+-\d+`")


def read_source(file: str, pr23: bool, ref: str) -> str | None:
    """Source text of `file` from the working tree, or from `ref` when `pr23`; None if absent."""
    if not pr23:
        path = REPO_ROOT / file
        return path.read_text(encoding="utf-8") if path.is_file() else None
    result = subprocess.run(
        ["git", "show", f"{ref}:{file}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return result.stdout.decode("utf-8") if result.returncode == 0 else None


def symbol_spans(source: str) -> dict[str, tuple[int, int]]:
    """Qualified name -> (first line, last line) for every def and class, nested ones dotted."""
    spans: dict[str, tuple[int, int]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = prefix + child.name
                spans[name] = (child.lineno, child.end_lineno or child.lineno)
                if isinstance(child, ast.ClassDef):
                    walk(child, name + ".")

    walk(ast.parse(source), "")
    return spans


def check_doc(text: str, pr23_ref: str) -> tuple[int, list[str], list[str]]:
    """Return (references checked, errors, warnings) for the doc `text`."""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    checked = 0

    def source_for(m: re.Match[str]) -> str | None:
        src = read_source(m["file"], bool(m["pr"]), pr23_ref)
        if src is None:
            msg = f"{m['file']}: not readable ({'pr23 ref ' + pr23_ref if m['pr'] else 'tree'})"
            (warnings if m["pr"] else errors).append(msg)
        return src

    for m in _SYMBOL_ROW.finditer(text):
        seen.add(m.group(0))
        src = source_for(m)
        if src is None:
            continue
        checked += 1
        span = symbol_spans(src).get(m["sym"])
        want = (int(m["a"]), int(m["b"]))
        if span != want:
            errors.append(f"{m['sym']} in {m['file']}: doc says {want}, code has {span}")
    for m in _ANCHOR_ROW.finditer(text):
        seen.add(m.group(0))
        src = source_for(m)
        if src is None:
            continue
        checked += 1
        lines = src.splitlines()
        a, b = int(m["a"]), int(m["b"])
        if not any(m["txt"] in line for line in lines[a - 1 : b]):
            errors.append(f"anchor {m['txt']!r} not within {m['file']}:{a}-{b}")
    covered = set()
    for token in seen:
        covered.update(_ANY_REF.findall(token))
    for ref in _ANY_REF.findall(text):
        if ref not in covered:
            errors.append(f"unchecked reference {ref} (use a symbol row or an anchor row)")
    return checked, errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc", nargs="?", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--pr23-ref", default=DEFAULT_PR23_REF)
    args = parser.parse_args(argv)
    checked, errors, warnings = check_doc(args.doc.read_text(encoding="utf-8"), args.pr23_ref)
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"checked {checked} references: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
