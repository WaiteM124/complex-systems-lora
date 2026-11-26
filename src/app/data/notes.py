"""Optional loader for private Markdown notes (tier N). Never committed."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.data.chunk import Chunk, chunk_text

WORKED_EXAMPLES_RE = re.compile(r"worked examples", re.IGNORECASE)
STUDY_RECORD_RE = re.compile(r"study\s*record", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class NotesDocument:
    path: Path
    text: str
    licence_class: str = "N"


def _drop_worked_example_sections(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    skipping = False
    skip_level = 0
    for line in lines:
        match = HEADING.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2)
            if WORKED_EXAMPLES_RE.search(title):
                skipping = True
                skip_level = level
                continue
            if skipping and level <= skip_level:
                skipping = False
        if skipping:
            continue
        out.append(line)
    return "\n".join(out).strip() + ("\n" if out else "")


def should_skip_notes_file(path: Path) -> bool:
    return bool(STUDY_RECORD_RE.search(path.name))


def load_notes_dir(directory: Path) -> list[NotesDocument]:
    if not directory.exists():
        return []
    docs: list[NotesDocument] = []
    for path in sorted(directory.rglob("*.md")):
        if should_skip_notes_file(path):
            continue
        text = _drop_worked_example_sections(path.read_text(encoding="utf-8"))
        if text.strip():
            docs.append(NotesDocument(path=path, text=text, licence_class="N"))
    return docs


def notes_to_chunks(docs: list[NotesDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_text(doc.text, source_id="notes"))
    # Retag section ids under notes/
    retagged: list[Chunk] = []
    for chunk in chunks:
        retagged.append(
            Chunk(
                section_path=chunk.section_path,
                section_id=chunk.section_id.replace("notes/", "notes/", 1)
                if chunk.section_id.startswith("notes/")
                else f"notes/{chunk.section_id.split('/', 1)[-1]}",
                text=chunk.text,
                token_count=chunk.token_count,
                chunk_hash=chunk.chunk_hash,
                chunk_index=chunk.chunk_index,
                source_id="notes",
            )
        )
    return retagged
