from __future__ import annotations

# List the unresolved {{NAME}} placeholders in the three deliverable drafts and check that every
# one is documented in report/PLACEHOLDERS.md (the table row starts with the placeholder), so
# filling them later is mechanical. A draft file that does not exist yet is reported, not an
# error: report/report.md may be committed after the others.
#
# Usage: python -m scripts.check_placeholders [--strict]
#   exit 1 when a placeholder is undocumented; with --strict also when any placeholder remains.
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_FILES: tuple[str, ...] = (
    "report/report.md",
    "report/model_card.md",
    "docs/WALKTHROUGH.md",
)
REGISTRY = "report/PLACEHOLDERS.md"

_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_REGISTRY_ROW = re.compile(r"^\s*\|\s*\{\{([A-Z][A-Z0-9_]*)\}\}\s*\|", re.MULTILINE)


def find_placeholders(text: str) -> list[str]:
    """Distinct placeholder names in `text`, in order of first appearance."""
    return list(dict.fromkeys(_PLACEHOLDER.findall(text)))


def documented_placeholders(registry_text: str) -> set[str]:
    """Names that have a table row in the registry (a mention in prose does not count)."""
    return set(_REGISTRY_ROW.findall(registry_text))


def check(root: Path = REPO_ROOT) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Return (placeholders per existing draft file, undocumented names, missing draft files)."""
    registry_path = root / REGISTRY
    documented = (
        documented_placeholders(registry_path.read_text(encoding="utf-8"))
        if registry_path.is_file()
        else set()
    )
    per_file: dict[str, list[str]] = {}
    missing: list[str] = []
    for rel in DRAFT_FILES:
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        per_file[rel] = find_placeholders(path.read_text(encoding="utf-8"))
    undocumented = sorted({n for names in per_file.values() for n in names} - documented)
    return per_file, undocumented, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail while any placeholder remains")
    args = parser.parse_args(argv)
    per_file, undocumented, missing = check()
    for rel, names in per_file.items():
        print(f"{rel}: {len(names)} unresolved")
        for name in names:
            print(f"  {{{{{name}}}}}")
    for rel in missing:
        print(f"{rel}: not present yet")
    for name in undocumented:
        print(f"UNDOCUMENTED {{{{{name}}}}} (add a row to {REGISTRY})")
    remaining = sum(len(v) for v in per_file.values())
    print(f"{remaining} unresolved in {len(per_file)} files, {len(undocumented)} undocumented")
    if undocumented or (args.strict and remaining):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
