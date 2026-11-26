"""Extract plain text from already-downloaded source files.

Supports arxiv_src (LaTeX), libretexts/html/jats (HTML), and pdf.
Inline maths is preserved as TeX (`$...$` or MathJax).
"""

from __future__ import annotations

import re
from pathlib import Path

MATH_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
SECTION_CMD = re.compile(r"\\(chapter|section|subsection|subsubsection)\*?\{([^}]*)\}")
LATEX_COMMENT = re.compile(r"(?m)^%.*$")
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")


def extract_file(path: Path, fmt: str) -> str:
    fmt_norm = fmt.lower()
    if fmt_norm == "arxiv_src":
        return extract_arxiv_src(path)
    if fmt_norm in {"libretexts", "html", "jats"}:
        return extract_html(path)
    if fmt_norm == "pdf":
        return extract_pdf(path)
    raise ValueError(f"unsupported extract format {fmt!r}")


def extract_arxiv_src(path: Path) -> str:
    """LaTeX → text, keeping `$...$` maths."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = LATEX_COMMENT.sub("", text)

    # Promote sectioning commands to markdown headings before stripping macros.
    def _heading(match: re.Match[str]) -> str:
        level = {"chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4}[match.group(1)]
        return f"\n{'#' * level} {match.group(2).strip()}\n"

    text = SECTION_CMD.sub(_heading, text)
    # Preserve math by shielding dollar spans, then strip remaining commands lightly.
    shields: list[str] = []

    def _shield(match: re.Match[str]) -> str:
        shields.append(match.group(0))
        return f"@@MATH{len(shields) - 1}@@"

    text = re.sub(r"\$\$?.+?\$\$?", _shield, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.DOTALL)
    text = LATEX_COMMAND.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    for index, chunk in enumerate(shields):
        text = text.replace(f"@@MATH{index}@@", chunk)
    return _collapse_blank_lines(text)


def extract_html(path: Path) -> str:
    """HTML → text; MathJax kept as TeX."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is required for HTML extraction; install the data extra"
        ) from exc

    # Convert MathJax delimiters to $...$ before stripping tags.
    raw = MATH_INLINE.sub(r"$\1$", raw)
    raw = MATH_DISPLAY.sub(r"$$\1$$", raw)
    raw = re.sub(
        r'<script[^>]*type=["\']math/tex[^"\']*["\'][^>]*>(.*?)</script>',
        lambda m: f"${m.group(1).strip()}$",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(heading.name[1])
        title = heading.get_text(" ", strip=True)
        heading.replace_with(soup.new_string(f"\n{'#' * level} {title}\n"))
    text = soup.get_text("\n")
    return _collapse_blank_lines(text)


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF extraction; install the data extra") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return _collapse_blank_lines("\n\n".join(pages))


def _collapse_blank_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip() + ("\n" if out else "")
