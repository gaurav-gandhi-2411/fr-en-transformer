from __future__ import annotations

from pathlib import Path

import pytest

from scripts import build_report_pdf as brp

SAMPLE = """# Title here

Intro with `code_id` and a [link text](https://example.org/x) and **bold** words.

| Set | Value |
|---|---|
| E1 | {{FINAL_E1_BLEU}} |

- first item
1. second item

```
fenced code
```
"""


def test_find_placeholders_dedupes_in_order() -> None:
    text = "{{A_ONE}} x {{B_TWO}} y {{A_ONE}} {single} {{lower}}"
    assert brp.find_placeholders(text) == ["{{A_ONE}}", "{{B_TWO}}"]


def test_rendered_text_drops_markup_but_keeps_visible_words() -> None:
    text = brp.rendered_text(SAMPLE)
    assert "https://example.org" not in text
    assert "link text" in text and "code_id" in text
    assert "---" not in text and "```" not in text and "fenced code" in text
    assert "**" not in text and "`" not in text


def test_count_words_with_and_without_placeholders() -> None:
    # Title here (2) + Intro with code_id and a link text and bold words (10) + Set Value E1 (3)
    # + first item (2) + second item (2) + fenced code (2, rendered) = 21; a placeholder adds one.
    assert brp.count_words(SAMPLE, count_placeholders=False) == 21
    assert brp.count_words(SAMPLE, count_placeholders=True) == 22


def test_count_words_ignores_punctuation_only_tokens() -> None:
    assert brp.count_words("one - two | three ... four") == 4


def test_check_page_limit_boundary() -> None:
    brp.check_page_limit(3, 3)
    with pytest.raises(brp.PageLimitError, match="4 pages, the limit is 3"):
        brp.check_page_limit(4, 3)


def test_check_strict_names_the_placeholders() -> None:
    brp.check_strict("no placeholders here")
    with pytest.raises(brp.PlaceholderError, match="FINAL_E1_BLEU"):
        brp.check_strict(SAMPLE)


def test_main_strict_fails_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "r.md"
    src.write_text(SAMPLE, encoding="utf-8")

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not render in --strict mode with placeholders left")

    monkeypatch.setattr(brp, "render_pdf", boom)
    assert brp.main(["--strict", "--src", str(src), "--out", str(tmp_path / "o.pdf")]) == 1


def test_main_fails_when_over_the_page_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "r.md"
    src.write_text("# T\n\nplain text\n", encoding="utf-8")
    monkeypatch.setattr(brp, "render_pdf", lambda _s, _o: None)
    monkeypatch.setattr(brp, "count_pdf_pages", lambda _p: 4)
    monkeypatch.setattr(brp, "pdf_text", lambda _p: "plain text")
    assert brp.main(["--src", str(src), "--out", str(tmp_path / "o.pdf")]) == 1
    assert "4 pages, the limit is 3" in capsys.readouterr().err
    monkeypatch.setattr(brp, "count_pdf_pages", lambda _p: 3)
    assert brp.main(["--src", str(src), "--out", str(tmp_path / "o.pdf")]) == 0


def test_main_reports_placeholders_in_non_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "r.md"
    src.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setattr(brp, "render_pdf", lambda _s, _o: None)
    monkeypatch.setattr(brp, "count_pdf_pages", lambda _p: 1)
    monkeypatch.setattr(brp, "pdf_text", lambda _p: "x")
    assert brp.main(["--src", str(src), "--out", str(tmp_path / "o.pdf")]) == 0
    out = capsys.readouterr().out
    assert "placeholders left: 1" in out
    assert "words (markdown, placeholders dropped): 21" in out
    assert "each placeholder counted as one word): 22" in out


@pytest.mark.skipif(
    not brp.toolchain_available(),
    reason="needs pandoc, a headless Edge/Chrome and pypdf (pypdf is not a project dependency)",
)
def test_render_small_sample_is_one_page_and_long_sample_overflows(tmp_path: Path) -> None:
    short = tmp_path / "short.md"
    short.write_text("# T\n\nA short paragraph.\n", encoding="utf-8")
    out = tmp_path / "short.pdf"
    brp.render_pdf(short, out)
    assert brp.count_pdf_pages(out) == 1

    long = tmp_path / "long.md"
    long.write_text("# T\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(120)))
    assert brp.main(["--src", str(long), "--out", str(tmp_path / "long.pdf")]) == 1
