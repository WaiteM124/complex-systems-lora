"""Section-aware chunking with rag-platform-compatible metadata fields."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.paths import repo_root

HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    section_path: str
    section_id: str
    text: str
    token_count: int
    chunk_hash: str
    chunk_index: int
    source_id: str


def count_tokens(text: str) -> int:
    """Whitespace/regex tokenizer — no model download."""
    return len(TOKEN_RE.findall(text))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_hash(text: str, section_path: str) -> str:
    return content_hash(f"{section_path}\n{text}")


@lru_cache(maxsize=1)
def load_data_config(path: Path | None = None) -> dict[str, Any]:
    data_path = path or (repo_root() / "config" / "data.yaml")
    raw: Any = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("data.yaml must be a mapping")
    return raw


def clear_data_config_cache() -> None:
    load_data_config.cache_clear()


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    lines = markdown.splitlines()
    blocks: list[tuple[str, str]] = []
    section_stack: list[str] = []
    buf: list[str] = []
    section_path = "document"

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            blocks.append((section_path, text))
        buf = []

    for line in lines:
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            section_stack = section_stack[: level - 1]
            section_stack.append(title)
            section_path = " › ".join(section_stack) if section_stack else "document"
            continue
        buf.append(line)
    flush()
    return blocks


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "section"


def chunk_text(
    text: str,
    *,
    source_id: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    config = load_data_config()
    chunk_cfg = config.get("chunk", {})
    lo = min_tokens if min_tokens is not None else int(chunk_cfg.get("min_tokens", 300))
    hi = max_tokens if max_tokens is not None else int(chunk_cfg.get("max_tokens", 500))
    if lo > hi:
        raise ValueError("min_tokens must be <= max_tokens")

    out: list[Chunk] = []
    idx = 0
    for section_path, block in _split_sections(text):
        section_id = f"{source_id}/{_slug(section_path)}"
        tokens = TOKEN_RE.findall(block)
        if len(tokens) <= hi:
            if len(tokens) < lo and len(tokens) > 0:
                # Keep short sections; callers may filter later.
                pass
            if not tokens:
                continue
            body = block.strip()
            prefixed = f"[{section_path}]\n{body}"
            out.append(
                Chunk(
                    section_path=section_path,
                    section_id=section_id,
                    text=prefixed,
                    token_count=count_tokens(prefixed),
                    chunk_hash=chunk_hash(body, section_path),
                    chunk_index=idx,
                    source_id=source_id,
                )
            )
            idx += 1
            continue

        start = 0
        while start < len(tokens):
            end = min(start + hi, len(tokens))
            # Prefer ending near min_tokens when possible for the last window.
            piece_tokens = tokens[start:end]
            piece = " ".join(piece_tokens).strip()
            if piece:
                prefixed = f"[{section_path}]\n{piece}"
                out.append(
                    Chunk(
                        section_path=section_path,
                        section_id=section_id,
                        text=prefixed,
                        token_count=count_tokens(prefixed),
                        chunk_hash=chunk_hash(piece, section_path),
                        chunk_index=idx,
                        source_id=source_id,
                    )
                )
                idx += 1
            if end >= len(tokens):
                break
            # Advance with modest overlap so chunks stay in [min, max].
            step = max(hi - max(lo // 5, 1), lo)
            start = min(start + step, end)
    return out
