"""Download raw sources into data/private/raw/.

Validates Content-Type and magic bytes against the declared format before
saving. Never call the live network helpers from tests — inject a transport.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.data.sources import Source, load_sources
from app.paths import repo_root

USER_AGENT = "complex-systems-lora-fetch/0.1 (+https://github.com/WaiteM124/complex-systems-lora)"

SUFFIX_FOR_FORMAT = {
    "arxiv_src": ".tex",
    "libretexts": ".html",
    "html": ".html",
    "jats": ".xml",
    "pdf": ".pdf",
    "pmc": ".html",
}

HttpGet = Callable[[str], tuple[bytes, str]]


class ContentMismatchError(ValueError):
    """Raised when downloaded bytes do not match the declared format."""


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    status: str  # ok | skipped | failed
    reason: str
    path: str | None = None


def raw_dir(private_root: Path | None = None) -> Path:
    root = private_root or (repo_root() / "data" / "private")
    path = root / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_http_get(url: str) -> tuple[bytes, str]:
    """Live HTTP GET. Not used in tests."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        content_type = response.headers.get("Content-Type", "") or ""
        return response.read(), content_type


def resolve_fetch_url(source: Source, url: str) -> str:
    """Rewrite arXiv abs URLs to the e-print endpoint."""
    if source.format == "arxiv_src" and "arxiv.org/abs/" in url:
        paper_id = url.rstrip("/").split("/")[-1]
        return f"https://arxiv.org/e-print/{paper_id}"
    return url


def _looks_like_html(data: bytes) -> bool:
    head = data.lstrip()[:200].lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or head.startswith(b"<head")
        or b"<html" in head[:100]
    )


def _looks_like_xml(data: bytes) -> bool:
    head = data.lstrip()[:200].lower()
    return head.startswith(b"<?xml") or head.startswith(b"<")


def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-" or data[:4] == b"%PDF"


def _looks_like_gzip(data: bytes) -> bool:
    return data[:2] == b"\x1f\x8b"


def _looks_like_tar(data: bytes) -> bool:
    # POSIX ustar magic at offset 257.
    return len(data) >= 265 and data[257:262] in {b"ustar", b"USTAR"}


def _looks_like_latex(data: bytes) -> bool:
    sample = data[:4000]
    # Tar/binary payloads contain NUL bytes; real .tex sources do not.
    if b"\x00" in sample or _looks_like_html(data) or _looks_like_tar(data):
        return False
    text = sample.decode("utf-8", errors="ignore")
    return bool(
        re.search(r"\\(documentclass|begin|section|chapter|usepackage)\b", text)
        or "\\title" in text
        or "\\maketitle" in text
    )


def validate_content(fmt: str, data: bytes, content_type: str) -> None:
    """Refuse to save when Content-Type / magic bytes disagree with format."""
    fmt_norm = fmt.lower()
    ctype = content_type.lower()

    if fmt_norm == "pdf":
        if "html" in ctype and "pdf" not in ctype:
            raise ContentMismatchError(f"Content-Type {content_type!r} is HTML, not PDF")
        if not _looks_like_pdf(data):
            raise ContentMismatchError(
                f"PDF magic missing (starts with {data[:16]!r}); refusing to save"
            )
        return

    if fmt_norm in {"html", "libretexts", "pmc"}:
        if not _looks_like_html(data) and not _looks_like_xml(data):
            raise ContentMismatchError(f"expected HTML for format {fmt_norm!r}, got {data[:16]!r}")
        return

    if fmt_norm == "jats":
        if _looks_like_html(data) and not data.lstrip().startswith(b"<?xml"):
            # Article landing pages are HTML; require XML/JATS.
            raise ContentMismatchError(
                "JATS/XML magic missing (got an HTML landing page); refusing to save"
            )
        if not _looks_like_xml(data):
            raise ContentMismatchError(f"expected XML/JATS, got {data[:16]!r}")
        return

    if fmt_norm == "arxiv_src":
        if _looks_like_html(data):
            raise ContentMismatchError("arXiv source looks like an HTML page; refusing to save")
        # Gzip/tar is unpacked before this check in normalise_payload.
        if not (_looks_like_latex(data) or _looks_like_gzip(data)):
            raise ContentMismatchError(f"expected LaTeX/gzip arXiv source, got {data[:16]!r}")
        return

    raise ContentMismatchError(f"unsupported format {fmt!r}")


def unpack_arxiv_payload(data: bytes) -> bytes:
    """Turn an arXiv e-print payload into a single LaTeX byte string."""
    payload = data
    if _looks_like_gzip(payload):
        payload = gzip.decompress(payload)

    # Prefer tar extraction when the payload is an archive (common for multi-file
    # e-prints). Checking latex first falsely matches .tex bytes inside a tar.
    if _looks_like_tar(payload):
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
                tex_members = [
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.lower().endswith(".tex")
                ]
                if not tex_members:
                    raise ContentMismatchError("arXiv archive contains no .tex file")
                # Prefer the largest .tex (usually the main paper).
                tex_members.sort(key=lambda member: member.size, reverse=True)
                extracted = archive.extractfile(tex_members[0])
                if extracted is None:
                    raise ContentMismatchError("failed to extract .tex from arXiv archive")
                return extracted.read()
        except tarfile.TarError as exc:
            raise ContentMismatchError(
                f"arXiv payload looks like tar but failed to open: {exc}"
            ) from exc

    if _looks_like_latex(payload):
        return payload

    raise ContentMismatchError(f"arXiv payload is not LaTeX or tar (starts with {payload[:16]!r})")


def normalise_payload(fmt: str, data: bytes, content_type: str) -> bytes:
    """Validate and normalise downloaded bytes for the declared format."""
    fmt_norm = fmt.lower()
    if fmt_norm == "arxiv_src":
        if _looks_like_html(data):
            raise ContentMismatchError("arXiv source looks like an HTML page; refusing to save")
        data = unpack_arxiv_payload(data)
        validate_content("arxiv_src", data, "text/plain")
        return data

    if fmt_norm == "html" and _looks_like_html(data) and len(data) < 5000:
        # SPA shells (e.g. networksciencebook.com/) are tiny HTML wrappers.
        # Multi-URL chapter fetches are concatenated and larger; allow those.
        body = data.lower()
        if b"ng-app" in body or b"ng-controller" in body:
            raise ContentMismatchError("HTML looks like an empty SPA shell, not chapter content")

    validate_content(fmt_norm, data, content_type)
    return data


def destination_for(source: Source, dest_dir: Path) -> Path:
    suffix = SUFFIX_FOR_FORMAT.get(source.format, ".bin")
    return dest_dir / f"{source.id}{suffix}"


def fetch_source(
    source: Source,
    *,
    dest_dir: Path,
    http_get: HttpGet | None = None,
) -> Path:
    """Download one ingestible source. Raises ContentMismatchError on bad bytes."""
    getter = http_get or default_http_get
    urls = source.download_urls()
    if not urls:
        raise ValueError(f"source {source.id} has no url")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = destination_for(source, dest_dir)
    parts: list[bytes] = []
    last_ctype = ""

    for index, raw_url in enumerate(urls):
        url = resolve_fetch_url(source, raw_url)
        data, content_type = getter(url)
        last_ctype = content_type
        if source.format == "html" and len(urls) > 1:
            # Chapter partials: wrap with a heading so chunking sees sections.
            heading = f"<h1>Chapter {index}</h1>\n".encode()
            # Validate each chapter looks like HTML content (not SPA shell).
            normalised = normalise_payload(source.format, data, content_type)
            parts.append(heading + normalised)
        else:
            parts.append(normalise_payload(source.format, data, content_type))

    if len(parts) == 1:
        payload = parts[0]
    else:
        payload = b"\n".join(parts)
        # Re-validate the concatenated HTML document.
        validate_content(source.format, payload, last_ctype or "text/html")

    dest.write_bytes(payload)
    return dest


def fetch_all(
    *,
    private_root: Path | None = None,
    http_get: HttpGet | None = None,
) -> list[FetchResult]:
    """Attempt every source in sources.yaml and return a per-source summary.

    Tier-C / ingest=false sources are skipped by design. Failures do not abort
    the rest of the run; each row records ok / skipped / failed.
    """
    dest = raw_dir(private_root)
    results: list[FetchResult] = []
    for source in load_sources().sources:
        if source.licence_class == "C" or not source.ingest:
            reason = (
                f"skipped by design (licence_class={source.licence_class}, ingest={source.ingest})"
            )
            results.append(FetchResult(source_id=source.id, status="skipped", reason=reason))
            continue
        try:
            path = fetch_source(source, dest_dir=dest, http_get=http_get)
            results.append(
                FetchResult(
                    source_id=source.id,
                    status="ok",
                    reason="downloaded and validated",
                    path=str(path),
                )
            )
        except ContentMismatchError as exc:
            results.append(
                FetchResult(
                    source_id=source.id,
                    status="failed",
                    reason=f"content mismatch: {exc}",
                )
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            results.append(
                FetchResult(
                    source_id=source.id,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def format_summary_table(results: list[FetchResult]) -> str:
    """Plain-text summary table for the CLI (Windows-safe newlines)."""
    col_id = max(len("id"), max(len(row.source_id) for row in results))
    col_status = max(len("status"), max(len(row.status) for row in results))
    header = f"{'id'.ljust(col_id)}  {'status'.ljust(col_status)}  reason"
    lines = [header, "-" * len(header)]
    for row in results:
        lines.append(f"{row.source_id.ljust(col_id)}  {row.status.ljust(col_status)}  {row.reason}")
    ok = sum(1 for row in results if row.status == "ok")
    skipped = sum(1 for row in results if row.status == "skipped")
    failed = sum(1 for row in results if row.status == "failed")
    lines.append("")
    lines.append(f"summary: ok={ok} skipped={skipped} failed={failed} total={len(results)}")
    return "\n".join(lines)
