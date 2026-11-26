from __future__ import annotations

from app.data.chunk import Chunk
from app.data.split import assign_sections, read_test_sections, split_chunks, write_test_sections


def _chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for index in range(10):
        section_id = f"fixture/s{index:02d}"
        out.append(
            Chunk(
                section_path=f"S{index}",
                section_id=section_id,
                text=f"[S{index}]\npassage {index} about systems",
                token_count=5,
                chunk_hash=str(index),
                chunk_index=index,
                source_id="fixture",
            )
        )
    return out


def test_split_deterministic() -> None:
    sections = [c.section_id for c in _chunks()]
    a = assign_sections(sections, seed=20261126)
    b = assign_sections(sections, seed=20261126)
    assert a == b


def test_no_section_in_two_splits() -> None:
    chunks = _chunks()
    assignment = assign_sections([c.section_id for c in chunks], seed=20261126)
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for section_id, split in assignment.items():
        by_split[split].add(section_id)
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    split = split_chunks(chunks, seed=20261126)
    seen: set[str] = set()
    for name in ("train", "val", "test"):
        for chunk in split[name]:
            assert chunk.section_id not in seen or assignment[chunk.section_id] == name
            seen.add(chunk.section_id)


def test_test_sections_file_matches(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "test_sections.txt"
    assignment = assign_sections([c.section_id for c in _chunks()], seed=20261126)
    write_test_sections(assignment, path)
    written = read_test_sections(path)
    expected = sorted(sid for sid, split in assignment.items() if split == "test")
    assert written == expected
