from __future__ import annotations

import csv
from pathlib import Path

from app.data.export_review import export_review_csv
from app.data.import_review import import_review_csv
from app.schema import Item


def _items() -> list[Item]:
    passage = "Resilience is the capacity to absorb disturbance and reorganise."
    return [
        Item.model_validate(
            {
                "id": "fixture-s01-q1",
                "source_id": "fixture",
                "licence_class": "A",
                "split": "test",
                "section_id": "fixture/s01",
                "passage": passage,
                "question": "What is resilience?",
                "target": {
                    "answerable": True,
                    "answer": "Capacity to absorb disturbance.",
                    "quote": "Resilience is the capacity to absorb disturbance and reorganise.",
                    "concepts": ["resilience"],
                },
            }
        ),
        Item.model_validate(
            {
                "id": "fixture-s01-q2",
                "source_id": "fixture",
                "licence_class": "A",
                "split": "test",
                "section_id": "fixture/s01",
                "passage": passage,
                "question": "What colour is the sky in the passage?",
                "target": {
                    "answerable": False,
                    "answer": "",
                    "quote": "",
                    "concepts": [],
                },
            }
        ),
        Item.model_validate(
            {
                "id": "fixture-s02-q1",
                "source_id": "fixture",
                "licence_class": "A",
                "split": "train",
                "section_id": "fixture/s02",
                "passage": "Feedback loops couple system variables.",
                "question": "What do feedback loops do?",
                "target": {
                    "answerable": True,
                    "answer": "They couple system variables.",
                    "quote": "Feedback loops couple system variables.",
                    "concepts": ["feedback-loop"],
                },
            }
        ),
    ]


def test_review_roundtrip_fix_and_drop(tmp_path: Path) -> None:
    items = _items()
    csv_path = tmp_path / "test_review.csv"
    export_review_csv(items, csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["id"] == "fixture-s01-q2":
                row["verdict"] = "drop"
            elif row["id"] == "fixture-s01-q1":
                row["verdict"] = "fix"
                row["fixed_answer"] = "It is the capacity to absorb disturbance and reorganise."
                row["fixed_quote"] = (
                    "Resilience is the capacity to absorb disturbance and reorganise."
                )
                row["fixed_concepts"] = "resilience"
            writer.writerow(row)

    result = import_review_csv(csv_path, existing=items)
    test_ids = {item.id for item in result if item.split == "test"}
    assert "fixture-s01-q2" not in test_ids
    fixed = next(item for item in result if item.id == "fixture-s01-q1")
    assert fixed.target.answer.startswith("It is the capacity")
    assert sum(1 for item in result if item.split == "train") == 1
