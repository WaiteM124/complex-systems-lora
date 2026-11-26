"""Export the test split to a CSV for hand review."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from app.schema import Item

REVIEW_COLUMNS = [
    "id",
    "source_id",
    "question",
    "passage",
    "answerable",
    "answer",
    "quote",
    "concepts",
    "verdict",
    "fixed_answer",
    "fixed_quote",
    "fixed_concepts",
    "notes",
]


def export_review_csv(items: Sequence[Item], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    test_items = [item for item in items if item.split == "test"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for item in test_items:
            writer.writerow(
                {
                    "id": item.id,
                    "source_id": item.source_id,
                    "question": item.question,
                    "passage": item.passage,
                    "answerable": item.target.answerable,
                    "answer": item.target.answer,
                    "quote": item.target.quote,
                    "concepts": "|".join(item.target.concepts),
                    "verdict": "ok",
                    "fixed_answer": "",
                    "fixed_quote": "",
                    "fixed_concepts": "",
                    "notes": "",
                }
            )
    return path
