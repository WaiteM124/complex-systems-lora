"""Apply hand-review CSV fixes/drops and re-run filters."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from app.data.filters import filter_items
from app.schema import Item, Target


def import_review_csv(
    path: Path,
    *,
    existing: Sequence[Item] | None = None,
) -> list[Item]:
    by_id = {item.id: item for item in (existing or [])}
    kept: list[Item] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            verdict = (row.get("verdict") or "ok").strip().lower()
            if verdict == "drop":
                continue
            item_id = row["id"]
            base = by_id.get(item_id)
            answerable = str(row.get("answerable", "")).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if verdict == "fix":
                answer = row.get("fixed_answer") or row.get("answer") or ""
                quote = row.get("fixed_quote") or row.get("quote") or ""
                concepts_raw = row.get("fixed_concepts") or row.get("concepts") or ""
            else:
                answer = row.get("answer") or ""
                quote = row.get("quote") or ""
                concepts_raw = row.get("concepts") or ""
            concepts = [c for c in concepts_raw.split("|") if c.strip()]
            if not answerable:
                answer, quote, concepts = "", "", []
            target = Target(
                answerable=answerable,
                answer=answer,
                quote=quote,
                concepts=concepts,
            )
            passage = row.get("passage") or (base.passage if base else "")
            split = base.split if base else "test"
            licence = base.licence_class if base else "A"
            section_id = base.section_id if base else f"{row['source_id']}/unknown"
            item = Item(
                id=item_id,
                source_id=row["source_id"],
                licence_class=licence,
                split=split,
                section_id=section_id,
                passage=passage,
                question=row.get("question") or "",
                target=target,
            )
            kept.append(item)
    # Preserve non-test items from existing.
    others = [item for item in (existing or []) if item.split != "test"]
    return filter_items([*others, *kept])
