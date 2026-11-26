from __future__ import annotations

from app.data.chunk import chunk_text, count_tokens
from app.data.extract import extract_file
from app.paths import repo_root

FIXTURES = repo_root() / "tests" / "fixtures"


def test_arxiv_keeps_inline_math() -> None:
    text = extract_file(FIXTURES / "arxiv_src" / "sample.tex", "arxiv_src")
    assert "$f$" in text or "$f(x^*)$" in text or "$" in text
    chunks = chunk_text(text, source_id="fixture", min_tokens=1, max_tokens=80)
    assert chunks
    for chunk in chunks:
        assert chunk.token_count == count_tokens(chunk.text)
        assert chunk.token_count <= 80 + 20  # prefix heading allowance


def test_libretexts_mathjax_survives() -> None:
    text = extract_file(FIXTURES / "libretexts" / "sample.html", "libretexts")
    assert "$" in text
    assert "Emergence" in text or "emergence" in text.lower()


def test_html_extract() -> None:
    text = extract_file(FIXTURES / "html" / "sample.html", "html")
    assert "small-world" in text.lower() or "Small-world" in text or "network" in text.lower()


def test_pdf_extract() -> None:
    text = extract_file(FIXTURES / "pdf" / "sample.pdf", "pdf")
    assert "Critical" in text or "critical" in text.lower() or "tipping" in text.lower()


def test_chunk_sizes_within_bounds() -> None:
    text = extract_file(FIXTURES / "arxiv_src" / "sample.tex", "arxiv_src")
    # Expand text so chunking must split.
    big = text + "\n\n" + ("word " * 400)
    chunks = chunk_text(big, source_id="fixture", min_tokens=50, max_tokens=120)
    assert chunks
    for chunk in chunks:
        # Allow the section prefix; body should stay near the max window.
        assert chunk.token_count <= 120 + 30


def test_jats_uses_xml_parser_without_warning() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = extract_file(FIXTURES / "jats" / "sample.xml", "jats")
    assert "## Introduction" in text
    assert "### Variance" in text
    assert "$\\rho_1$" in text
    assert "$\\lambda$" in text
    assert "Synthetic reference" not in text
    assert "## Figure title" not in text
    chunks = chunk_text(text, source_id="fixture", min_tokens=1, max_tokens=80)
    assert any("Variance" in chunk.section_path for chunk in chunks)


def test_jats_falls_back_to_html_for_html_files() -> None:
    text = extract_file(FIXTURES / "html" / "sample.html", "jats")
    assert "small-world" in text.lower()
