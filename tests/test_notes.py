from __future__ import annotations

from pathlib import Path

from app.data.licence import route_item
from app.data.notes import load_notes_dir, notes_to_chunks, should_skip_notes_file
from app.paths import repo_root
from app.schema import Item


def test_worked_examples_dropped_and_study_record_skipped(tmp_path: Path) -> None:
    src = repo_root() / "tests" / "fixtures" / "notes"
    # Copy into tmp to avoid relying on write permissions elsewhere
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    for path in src.glob("*"):
        (notes_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    assert should_skip_notes_file(notes_dir / "study record.md")
    docs = load_notes_dir(notes_dir)
    assert docs
    joined = "\n".join(doc.text for doc in docs)
    assert "worked example" not in joined.lower()
    assert all(doc.licence_class == "N" for doc in docs)
    assert not any(should_skip_notes_file(doc.path) for doc in docs)

    chunks = notes_to_chunks(docs)
    assert chunks
    assert all(chunk.source_id == "notes" for chunk in chunks)

    hierarchy = next(c for c in chunks if "Hierarchy organises" in c.text)
    quote = "Hierarchy organises subsystems across scales."
    item = Item.model_validate(
        {
            "id": "notes-s01-q1",
            "source_id": "notes",
            "licence_class": "N",
            "split": "train",
            "section_id": hierarchy.section_id,
            "passage": hierarchy.text,
            "question": "What is hierarchy?",
            "target": {
                "answerable": True,
                "answer": quote,
                "quote": quote,
                "concepts": ["hierarchy"],
            },
        }
    )
    path = route_item(item, private_root=tmp_path / "private", public_root=tmp_path)
    assert "private" in str(path)
    assert path.name.startswith("n_")
