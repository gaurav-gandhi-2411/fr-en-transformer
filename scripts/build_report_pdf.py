from __future__ import annotations

# Render report/report.md to report/build/report.pdf and enforce the page limit.
#
# Toolchain (no global installs, nothing downloaded): pandoc turns the markdown into a standalone
# HTML page with the print CSS below (A4, 1.8 cm margins, 10 pt body, 8 pt tables), and a local
# Chromium-family browser (Edge or Chrome, headless) prints it to PDF. Chosen over pandoc +
# tectonic because tectonic downloads TeX packages on first use, and over weasyprint because it
# needs GTK on Windows. The page count is read back with pypdf, which is NOT a project dependency
# (uv.lock is unchanged): install it in a throwaway venv to build, e.g. `uv pip install pypdf`.
#
# The PDF is a submission artifact for the company, not for the repo: report/build/ is gitignored.
#
# Usage: python -m scripts.build_report_pdf [--strict] [--max-pages 3] [--src FILE] [--out FILE]
#   default (non-strict): placeholders `{{NAME}}` render as-is so the draft can be sized.
#   --strict: exit 1 before rendering if any placeholder remains.
#   exit 1 when the PDF has more than --max-pages pages.
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "report" / "report.md"
DEFAULT_OUT = REPO_ROOT / "report" / "build" / "report.pdf"
MAX_PAGES = 3

PLACEHOLDER_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")

# Print stylesheet. Sizes are the typography budget: 10 pt body, 8 pt tables, 1.8 cm margins.
CSS = """
@page { size: A4; margin: 1.8cm; }
html { font-family: Calibri, "Segoe UI", "Liberation Sans", Arial, sans-serif; font-size: 10pt;
       line-height: 1.22; hyphens: auto; color: #111; }
/* pandoc -s injects a 36em-wide centred body and scrolling tables: undo both for print */
body { margin: 0; padding: 0; max-width: none; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; line-height: 1.15; }
h2 { font-size: 11.5pt; margin: 8pt 0 3pt 0; }
p { margin: 0 0 3pt 0; text-align: left; }
ol, ul { margin: 0 0 4pt 0; padding-left: 16pt; }
li { margin: 0; }
ol { font-size: 8.5pt; line-height: 1.15; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 8.5pt;
       white-space: normal; overflow-wrap: anywhere; }
table { display: table; overflow: visible; border-collapse: collapse; width: 100%;
        font-size: 8pt; line-height: 1.18; margin: 2pt 0 4pt 0; }
td code, th code { font-size: 7pt; }
th, td { border: 0.4pt solid #888; padding: 1.5pt 3pt; vertical-align: top; text-align: left; }
th { background: #eee; }
tr { page-break-inside: avoid; }
header, #title-block-header { display: none; }
"""


class PageLimitError(RuntimeError):
    """The rendered PDF has more pages than allowed."""


class PlaceholderError(RuntimeError):
    """Unresolved {{...}} placeholders remain in --strict mode."""


def find_placeholders(text: str) -> list[str]:
    """Distinct `{{NAME}}` placeholders in `text`, in order of first appearance."""
    return list(dict.fromkeys(PLACEHOLDER_RE.findall(text)))


def rendered_text(md: str) -> str:
    """Approximate the visible text of the markdown: no fence lines, link targets, table rules,
    list markers, heading hashes, emphasis stars or code backticks. Fenced code is kept (it
    renders). Placeholders are kept as tokens."""
    out: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*", line):
                continue  # table separator row
            line = re.sub(r"^\s*#{1,6}\s+", "", line)
            line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line)
            line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)
            line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
            line = line.replace("|", " ").replace("*", "").replace("`", "")
        out.append(line)
    return "\n".join(out)


def count_words(md: str, count_placeholders: bool = True) -> int:
    """Whitespace-separated tokens of the rendered text that contain a letter or digit.

    With `count_placeholders` each `{{NAME}}` counts as one word; without it they are dropped,
    which is the size the filled report will approach (a filled value is at least one word).
    """
    text = rendered_text(md)
    if not count_placeholders:
        text = PLACEHOLDER_RE.sub(" ", text)
    return sum(1 for tok in text.split() if re.search(r"[A-Za-z0-9]", tok))


def check_page_limit(pages: int, max_pages: int = MAX_PAGES) -> None:
    """Raise PageLimitError with an actionable message when `pages` exceeds `max_pages`."""
    if pages > max_pages:
        raise PageLimitError(
            f"report.pdf has {pages} pages, the limit is {max_pages}: trim the text in "
            "report/report.md (not the evidence), then rebuild"
        )


def check_strict(md: str) -> None:
    """Raise PlaceholderError if any placeholder is left (used by --strict)."""
    left = find_placeholders(md)
    if left:
        raise PlaceholderError(f"{len(left)} unresolved placeholders: " + ", ".join(left))


def find_browser() -> str | None:
    """A Chromium-family browser: $REPORT_BROWSER, then PATH, then the usual Windows locations."""
    env = os.environ.get("REPORT_BROWSER")
    if env and Path(env).is_file():
        return env
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    # Windows spells these variables in mixed case; os.environ is case-insensitive there.
    for var in ("ProgramFiles(x86)", "ProgramFiles"):  # noqa: SIM112
        base = os.environ.get(var)
        if not base:
            continue
        for rel in (
            "Microsoft/Edge/Application/msedge.exe",
            "Google/Chrome/Application/chrome.exe",
        ):
            if (Path(base) / rel).is_file():
                return str(Path(base) / rel)
    return None


def toolchain_available() -> bool:
    """True when pandoc, a headless browser and pypdf are all present."""
    try:
        import pypdf  # noqa: F401  (availability probe only)
    except ImportError:
        return False
    return shutil.which("pandoc") is not None and find_browser() is not None


def count_pdf_pages(pdf: Path) -> int:
    """Number of pages in `pdf`, read with pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # fail closed: never guess a page count
        raise RuntimeError(
            "pypdf is needed to count pages; install it in a throwaway venv (not a project "
            "dependency)"
        ) from exc
    return len(PdfReader(str(pdf)).pages)


def pdf_text(pdf: Path) -> str:
    """Extracted text of every page of `pdf` (pypdf), for the rendered-word count."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)


def render_pdf(src: Path, out: Path) -> None:
    """markdown -> HTML (pandoc) -> PDF (headless browser). Raises RuntimeError on tool failure."""
    pandoc = shutil.which("pandoc")
    browser = find_browser()
    if pandoc is None or browser is None:
        raise RuntimeError("pandoc and a Chromium-family browser (Edge or Chrome) are required")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "style.html").write_text(f"<style>{CSS}</style>\n", encoding="utf-8")
        html = tmp / "report.html"
        subprocess.run(
            [
                pandoc,
                str(src),
                "-f",
                "markdown+pipe_tables",
                "-t",
                "html5",
                "-s",
                "-V",
                "lang=en",
                "--metadata",
                "pagetitle=report",
                "-H",
                str(tmp / "style.html"),
                "-o",
                str(html),
            ],  # fmt: skip
            check=True,
            capture_output=True,
        )
        out.unlink(missing_ok=True)
        subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={tmp / 'profile'}",
                f"--print-to-pdf={out}",
                html.as_uri(),
            ],  # fmt: skip
            check=True,
            capture_output=True,
            timeout=120,
        )
    if not out.is_file():
        raise RuntimeError("the browser exited without writing the PDF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--strict", action="store_true", help="fail while any {{...}} remains")
    args = parser.parse_args(argv)

    md = args.src.read_text(encoding="utf-8")
    left = find_placeholders(md)
    try:
        if args.strict:
            check_strict(md)
        render_pdf(args.src, args.out)
        pages = count_pdf_pages(args.out)
    except (PlaceholderError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    words_no_ph = count_words(md, count_placeholders=False)
    words_ph = count_words(md, count_placeholders=True)
    print(f"pages: {pages} (limit {args.max_pages})")
    print(f"words (markdown, placeholders dropped): {words_no_ph}")
    print(f"words (markdown, each placeholder counted as one word): {words_ph}")
    print(f"words (extracted from the PDF text): {len(pdf_text(args.out).split())}")
    print(f"placeholders left: {len(left)}")
    try:
        check_page_limit(pages, args.max_pages)
    except PageLimitError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
