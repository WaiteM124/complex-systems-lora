from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import cmd_chunk
from app.data.sources import clear_sources_cache


class _Args:
    no_env = True


def test_chunk_skips_bad_pdf_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A HTML landing page saved as .pdf must not abort the whole chunk run."""
    clear_sources_cache()
    private = tmp_path / "private"
    raw = private / "raw"
    raw.mkdir(parents=True)

    # Bad PDF for Think Complexity (the Windows failure mode).
    (raw / "cs-041.pdf").write_bytes(b"<!DOCTYPE html><html><body>landing</body></html>")

    # Good HTML for Network Science so the run still produces chunks.
    good_html = (
        "<html><body><h1>Networks</h1><p>"
        + ("scale-free preferential attachment " * 80)
        + "</p><h2>Robustness</h2><p>"
        + ("hubs fail randomly " * 80)
        + "</p></body></html>"
    )
    (raw / "cs-021.html").write_text(good_html, encoding="utf-8")

    # Good arXiv-like tex for a tier-A source.
    (raw / "cs-033.tex").write_text(
        "\\section{Utility}\nA fixed point satisfies $f(x)=x$.\n"
        + ("stability analysis words " * 100)
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("PRIVATE_DATA_DIR", str(private))

    code = cmd_chunk(_Args())  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "WARNING: cs-041:" in captured.err
    assert code == 0

    chunks_path = private / "chunks.jsonl"
    assert chunks_path.is_file()
    lines = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    source_ids = {row["source_id"] for row in lines}
    assert "cs-041" not in source_ids
    assert "cs-021" in source_ids or "cs-033" in source_ids


def test_chunk_exits_nonzero_when_nothing_chunked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_sources_cache()
    private = tmp_path / "private"
    (private / "raw").mkdir(parents=True)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("PRIVATE_DATA_DIR", str(private))
    code = cmd_chunk(_Args())  # type: ignore[arg-type]
    assert code == 1
