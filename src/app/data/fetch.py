"""Download raw sources into data/private/raw/. Never call from tests."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from app.data.sources import Source, ingestable_sources
from app.paths import repo_root


def raw_dir(private_root: Path | None = None) -> Path:
    root = private_root or (repo_root() / "data" / "private")
    path = root / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_source(source: Source, *, dest_dir: Path) -> Path:
    if not source.url:
        raise ValueError(f"source {source.id} has no url")
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = {
        "arxiv_src": ".tex",
        "libretexts": ".html",
        "html": ".html",
        "jats": ".xml",
        "pdf": ".pdf",
        "pmc": ".html",
    }.get(source.format, ".bin")
    dest = dest_dir / f"{source.id}{suffix}"
    # For arXiv abs URLs, prefer e-print source when possible.
    url = source.url
    if source.format == "arxiv_src" and "arxiv.org/abs/" in url:
        paper_id = url.rstrip("/").split("/")[-1]
        url = f"https://arxiv.org/e-print/{paper_id}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — operator-run fetch
        dest.write_bytes(response.read())
    return dest


def fetch_all(*, private_root: Path | None = None) -> list[Path]:
    dest = raw_dir(private_root)
    paths: list[Path] = []
    for source in ingestable_sources():
        paths.append(fetch_source(source, dest_dir=dest))
    return paths
