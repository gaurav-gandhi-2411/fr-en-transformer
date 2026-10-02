from __future__ import annotations

from pathlib import Path

from scripts import check_placeholders as cp


def _tree(tmp_path: Path, registry: str, files: dict[str, str]) -> Path:
    (tmp_path / "report").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / cp.REGISTRY).write_text(registry, encoding="utf-8")
    for rel, text in files.items():
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return tmp_path


def test_find_placeholders_dedupes_and_ignores_lowercase_and_single_braces() -> None:
    text = "a {{FINAL_E1_BLEU}} b {{FINAL_E1_BLEU}} {{GAPV2_X}} {{lower}} {not} {{9BAD}}"
    assert cp.find_placeholders(text) == ["FINAL_E1_BLEU", "GAPV2_X"]


def test_registry_row_counts_but_prose_mention_does_not() -> None:
    registry = "| {{A_ONE}} | x |\nsee also {{B_TWO}} in prose\n  | {{C_THREE}} | y |\n"
    assert cp.documented_placeholders(registry) == {"A_ONE", "C_THREE"}


def test_check_reports_undocumented_and_missing_files(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        "| {{KNOWN}} | x |\n",
        {"report/model_card.md": "{{KNOWN}} {{FORGOTTEN}}", "docs/WALKTHROUGH.md": "none"},
    )
    per_file, undocumented, missing = cp.check(root)
    assert per_file["report/model_card.md"] == ["KNOWN", "FORGOTTEN"]
    assert per_file["docs/WALKTHROUGH.md"] == []
    assert undocumented == ["FORGOTTEN"]
    assert missing == ["report/report.md"]


def test_real_drafts_document_every_placeholder() -> None:
    per_file, undocumented, _missing = cp.check()
    # report/report.md may be committed separately; the other two drafts must exist.
    assert "report/model_card.md" in per_file
    assert "docs/WALKTHROUGH.md" in per_file
    assert undocumented == []
