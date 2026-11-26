from __future__ import annotations

from pathlib import Path

import pytest

from app.data.fetch import (
    ContentMismatchError,
    fetch_all,
    fetch_source,
    format_summary_table,
    normalise_payload,
    unpack_arxiv_payload,
    validate_content,
)
from app.data.sources import Source, clear_sources_cache, load_sources


def _pdf_bytes() -> bytes:
    # Minimal %PDF header; enough for magic-byte checks.
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def _html_bytes(body: str = "<html><body><p>Hello</p></body></html>") -> bytes:
    return body.encode("utf-8")


def test_validate_pdf_rejects_html_landing_page() -> None:
    with pytest.raises(ContentMismatchError):
        validate_content("pdf", _html_bytes("<!DOCTYPE html><html></html>"), "text/html")


def test_validate_pdf_accepts_magic() -> None:
    validate_content("pdf", _pdf_bytes(), "application/pdf")


def test_validate_jats_rejects_html_article_page() -> None:
    with pytest.raises(ContentMismatchError):
        validate_content(
            "jats",
            _html_bytes("<!DOCTYPE html><html><body>article</body></html>"),
            "text/html",
        )


def test_fetch_source_refuses_to_save_mismatch(tmp_path: Path) -> None:
    source = Source(
        id="cs-041",
        title="Think Complexity fixture",
        licence_class="B",
        format="pdf",
        ingest=True,
        url="https://example.invalid/landing",
    )

    def fake_get(_url: str) -> tuple[bytes, str]:
        return _html_bytes("<!DOCTYPE html><html>landing</html>"), "text/html"

    with pytest.raises(ContentMismatchError):
        fetch_source(source, dest_dir=tmp_path, http_get=fake_get)
    assert list(tmp_path.glob("cs-041.*")) == []


def test_fetch_source_saves_valid_pdf(tmp_path: Path) -> None:
    source = Source(
        id="cs-041",
        title="Think Complexity fixture",
        licence_class="B",
        format="pdf",
        ingest=True,
        url="https://example.invalid/book.pdf",
    )

    def fake_get(_url: str) -> tuple[bytes, str]:
        return _pdf_bytes(), "application/pdf"

    path = fetch_source(source, dest_dir=tmp_path, http_get=fake_get)
    assert path.name == "cs-041.pdf"
    assert path.read_bytes().startswith(b"%PDF")


def test_fetch_source_concatenates_html_chapters(tmp_path: Path) -> None:
    source = Source(
        id="cs-021",
        title="Network Science fixture",
        licence_class="B",
        format="html",
        ingest=True,
        url="https://example.invalid/",
        urls=[
            "https://example.invalid/ch-0.html",
            "https://example.invalid/ch-1.html",
        ],
    )
    bodies = {
        "https://example.invalid/ch-0.html": _html_bytes(
            "<section><h2>Intro</h2><p>" + ("word " * 200) + "</p></section>"
        ),
        "https://example.invalid/ch-1.html": _html_bytes(
            "<section><h2>Graph</h2><p>" + ("node " * 200) + "</p></section>"
        ),
    }

    def fake_get(url: str) -> tuple[bytes, str]:
        return bodies[url], "text/html"

    path = fetch_source(source, dest_dir=tmp_path, http_get=fake_get)
    text = path.read_text(encoding="utf-8")
    assert "Chapter 0" in text and "Chapter 1" in text
    assert "Intro" in text and "Graph" in text


def test_fetch_all_summary_skips_tier_c(tmp_path: Path) -> None:
    clear_sources_cache()
    sources = load_sources()
    assert len(sources.sources) == 46

    def fake_get(url: str) -> tuple[bytes, str]:
        if url.endswith(".pdf") or "thinkcomplexity" in url:
            return _pdf_bytes(), "application/pdf"
        if "manuscript" in url or url.endswith(".xml"):
            return b'<?xml version="1.0"?><article><body>x</body></article>', "text/xml"
        if "e-print" in url or "arxiv.org" in url:
            # Minimal latex-looking payload (already unpacked form).
            return b"\\documentclass{article}\\begin{document}Hi\\end{document}", "application/gzip"
        # HTML chapters / libretexts
        return _html_bytes("<html><body><p>" + ("alpha " * 300) + "</p></body></html>"), "text/html"

    results = fetch_all(private_root=tmp_path, http_get=fake_get)
    assert len(results) == 46
    skipped = [row for row in results if row.status == "skipped"]
    assert len(skipped) == 40
    assert all("licence_class=C" in row.reason or "ingest=False" in row.reason for row in skipped)
    table = format_summary_table(results)
    assert "cs-001" in table and "skipped" in table
    assert "summary:" in table


def test_unpack_arxiv_gzip_tex() -> None:
    import gzip

    tex = b"\\documentclass{article}\\begin{document}Hello\\end{document}\n"
    payload = gzip.compress(tex)
    out = unpack_arxiv_payload(payload)
    assert b"\\documentclass" in out
    normalise_payload("arxiv_src", payload, "application/gzip")


def test_unpack_arxiv_gzip_tar_prefers_main_tex() -> None:
    import gzip
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        small = b"\\section{Appendix}\nshort\n"
        large = b"\\documentclass{article}\\begin{document}Main paper body\\end{document}\n"
        for name, body in (("notes.tex", small), ("main.tex", large), ("acl.sty", b"% sty")):
            info = tarfile.TarInfo(name=name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    payload = gzip.compress(buf.getvalue())
    out = unpack_arxiv_payload(payload)
    assert out.startswith(b"\\documentclass")
    assert b"Main paper body" in out
