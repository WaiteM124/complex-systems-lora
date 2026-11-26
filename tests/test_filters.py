from __future__ import annotations

from app.data.filters import deduplicate, enforce_answerable_ratio, filter_items
from app.schema import Item


def _item(
    item_id: str,
    *,
    question: str,
    answerable: bool = True,
    quote: str | None = None,
    passage: str = "Alpha beta gamma delta epsilon zeta eta theta.",
    answer: str = "Alpha beta.",
    concepts: list[str] | None = None,
) -> Item:
    if not answerable:
        target = {
            "answerable": False,
            "answer": "",
            "quote": "",
            "concepts": [],
        }
    else:
        target = {
            "answerable": True,
            "answer": answer,
            "quote": quote
            if quote is not None
            else "Alpha beta gamma delta epsilon zeta eta theta.",
            "concepts": concepts or ["complex-system"],
        }
    return Item.model_validate(
        {
            "id": item_id,
            "source_id": "fixture",
            "licence_class": "A",
            "split": "train",
            "section_id": "fixture/s01",
            "passage": passage,
            "question": question,
            "target": target,
        }
    )


def test_verbatim_quote_filter() -> None:
    good = _item("g1", question="q1")
    bad_raw = good.model_dump()
    bad_raw["target"]["quote"] = "not in passage"
    filtered = filter_items([good.model_dump(), bad_raw])
    assert [item.id for item in filtered] == ["g1"]


def test_dedup_near_duplicates() -> None:
    a = _item("a", question="What is alpha beta gamma delta epsilon?")
    b = _item("b", question="What is alpha beta gamma delta epsilon?")
    kept = deduplicate([a, b])
    assert len(kept) == 1


def test_answerable_ratio() -> None:
    answerable = [
        _item(f"a{i}", question=f"question answerable {i} unique words here") for i in range(8)
    ]
    unanswerable = [
        _item(
            f"u{i}",
            question=f"question unanswerable {i} unique words here",
            answerable=False,
        )
        for i in range(8)
    ]
    kept = enforce_answerable_ratio([*answerable, *unanswerable], unanswerable_share=0.2)
    n_unans = sum(1 for item in kept if not item.target.answerable)
    n_ans = sum(1 for item in kept if item.target.answerable)
    assert n_ans == 8
    assert n_unans == 2
