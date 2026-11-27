"""Synthetic CC0 items for tests (short invented text, no corpus material)."""

from __future__ import annotations

import json
from pathlib import Path

from app.schema import Item, Target

PASSAGES = [
    "A fixed point of a map satisfies f(x)=x. Linear stability depends on the derivative.",
    "Critical slowing down appears near a tipping point. Variance rises before the shift.",
    "A limit cycle is an isolated closed orbit. Nearby orbits spiral towards it over time.",
    "Feedback loops can amplify or damp change. Negative feedback tends to stabilise.",
]
CONCEPTS = [
    ["fixed-point", "linear-stability"],
    ["tipping-point", "critical-slowing-down"],
    ["limit-cycle"],
    ["feedback-loop"],
]


def make_item(index: int, split: str, *, answerable: bool = True) -> Item:
    passage_index = index % len(PASSAGES)
    passage = f"[{split} {index}] " + PASSAGES[passage_index]
    if answerable:
        sentence = PASSAGES[passage_index].split(". ")[0] + "."
        target = Target(
            answerable=True,
            answer=f"Answer {index}.",
            quote=sentence,
            concepts=CONCEPTS[passage_index],
        )
    else:
        target = Target(answerable=False)
    return Item(
        id=f"syn-{split}-{index:03d}",
        source_id="fixture",
        licence_class="A",
        split=split,  # type: ignore[arg-type]
        section_id=f"fixture/{split}-{index:03d}",
        passage=passage,
        question=f"Synthetic question {index}?",
        target=target,
        source="fixture",
    )


def make_items(n_train: int = 8, n_val: int = 4, n_test: int = 5) -> list[Item]:
    items: list[Item] = []
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for index in range(count):
            items.append(make_item(index, split, answerable=index % 4 != 3))
    return items


def write_items(path: Path, items: list[Item]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")
    return path


def answer_text(item: Item) -> str:
    return json.dumps(item.target.model_dump())
