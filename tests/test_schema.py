from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schema import Item, Target


def _valid_item(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "cs-018-s03-p002-q1",
        "source_id": "fixture",
        "licence_class": "A",
        "split": "train",
        "section_id": "fixture/s03",
        "passage": "A fixed point satisfies f(x)=x. Stability follows from the derivative.",
        "question": "What is a fixed point?",
        "target": {
            "answerable": True,
            "answer": "A fixed point satisfies f(x)=x.",
            "quote": "A fixed point satisfies f(x)=x.",
            "concepts": ["fixed-point"],
        },
    }
    data.update(overrides)
    return data


def test_valid_item() -> None:
    item = Item.model_validate(_valid_item())
    assert item.target.answerable is True


def test_quote_not_substring_rejected() -> None:
    raw = _valid_item()
    target = dict(raw["target"])  # type: ignore[arg-type]
    target["quote"] = "this quote is not in the passage"
    raw["target"] = target
    with pytest.raises(ValidationError):
        Item.model_validate(raw)


def test_quote_three_sentences_rejected() -> None:
    passage = "One. Two. Three. Four."
    raw = _valid_item(passage=passage)
    target = dict(raw["target"])  # type: ignore[arg-type]
    target["quote"] = "One. Two. Three."
    raw["target"] = target
    with pytest.raises(ValidationError):
        Item.model_validate(raw)


def test_unknown_concept_rejected() -> None:
    raw = _valid_item()
    target = dict(raw["target"])  # type: ignore[arg-type]
    target["concepts"] = ["not-a-real-tag"]
    raw["target"] = target
    with pytest.raises(ValidationError):
        Item.model_validate(raw)


def test_licence_c_rejected() -> None:
    with pytest.raises(ValidationError):
        Item.model_validate(_valid_item(licence_class="C"))


def test_unanswerable_nonempty_quote_rejected() -> None:
    with pytest.raises(ValidationError):
        Target(answerable=False, answer="", quote="leftover", concepts=[])
