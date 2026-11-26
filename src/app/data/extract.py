"""Extract plain text from already-downloaded source files.

Supports arxiv_src (LaTeX), libretexts/html (HTML), jats (XML), and pdf.
Inline maths is preserved as TeX (`$...$` or MathJax).
"""

from __future__ import annotations

import re
from pathlib import Path

MATH_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
SECTION_CMD = re.compile(r"\\(chapter|section|subsection|subsubsection)\*?\{([^}]*)\}")
LATEX_COMMENT = re.compile(r"(?m)^%.*$")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:])(?=\s|$)")
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")


def extract_file(path: Path, fmt: str) -> str:
    fmt_norm = fmt.lower()
    if fmt_norm == "arxiv_src":
        return extract_arxiv_src(path)
    if fmt_norm == "jats":
        return extract_jats(path)
    if fmt_norm in {"libretexts", "html"}:
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


def _looks_like_html_document(raw: str) -> bool:
    head = raw.lstrip()[:2000].lower()
    if head.startswith("<?xml") or "<article" in head:
        return False
    return "<html" in head or "<!doctype html" in head


def extract_jats(path: Path) -> str:
    """JATS XML → text with the XML parser; headings from <sec><title>, maths as TeX.

    Falls back to HTML extraction when the file is actually an HTML page.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if _looks_like_html_document(raw):
        return extract_html(path)
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is required for JATS extraction; install the data extra"
        ) from exc

    soup = BeautifulSoup(raw, features="xml")
    for tag in soup.find_all(["ref-list", "object-id", "graphic", "permissions"]):
        tag.decompose()
    # Formulas: prefer the TeX form, else MathML alttext, else its text.
    for formula in soup.find_all(["inline-formula", "disp-formula"]):
        tex = formula.find("tex-math")
        math = formula.find("math")
        if tex is not None:
            body = tex.get_text(" ", strip=True)
        elif math is not None:
            body = str(math.get("alttext") or math.get_text(" ", strip=True))
        else:
            body = formula.get_text(" ", strip=True)
        body = body.strip().strip("$").strip()
        delim = "$$" if formula.name == "disp-formula" else "$"
        formula.replace_with(soup.new_string(f" {delim}{body}{delim} "))
    for title in soup.find_all("title"):
        parent = title.parent
        if parent is not None and parent.name == "sec":
            level = min(6, 1 + len(title.find_parents("sec")))
            text = " ".join(title.get_text(" ").split())
            title.replace_with(soup.new_string(f"\n{'#' * level} {text}\n"))
    for para in soup.find_all("p"):
        text = " ".join(para.get_text(" ").split())
        para.replace_with(soup.new_string(f"\n{text}\n"))

    parts: list[str] = []
    article_title = soup.find("article-title")
    if article_title is not None:
        parts.append(f"# {' '.join(article_title.get_text(' ').split())}")
    abstract = soup.find("abstract")
    if abstract is not None:
        parts.append("## Abstract\n" + abstract.get_text(""))
    body_tag = soup.find("body")
    if body_tag is not None:
        parts.append(body_tag.get_text(""))
    if not parts:
        parts.append(soup.get_text("\n"))
    lines = [
        SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(line.split()))
        for line in "\n".join(parts).splitlines()
    ]
    return _collapse_blank_lines("\n".join(lines))


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
