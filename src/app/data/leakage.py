"""Fail if any test passage shares a 13-gram with train or validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.data.filters import ngrams, normalize_text
from app.schema import Item


class LeakageError(ValueError):
    """Raised when train/val and test passages leak n-grams."""


def passage_ngrams(passage: str, n: int = 13) -> set[str]:
    return ngrams(passage, n)


def find_leaks(
    train_val_passages: Sequence[str],
    test_passages: Sequence[str],
    *,
    n: int = 13,
) -> list[tuple[str, str, str]]:
    """Return list of (test_passage_snippet, train_passage_snippet, shared_ngram)."""
    index: dict[str, str] = {}
    for passage in train_val_passages:
        for gram in passage_ngrams(passage, n):
            if gram and gram not in index:
                index[gram] = passage
    leaks: list[tuple[str, str, str]] = []
    for passage in test_passages:
        for gram in passage_ngrams(passage, n):
            if gram in index:
                leaks.append((passage[:120], index[gram][:120], gram))
                break
    return leaks


def assert_no_leakage(
    train_val_passages: Sequence[str],
    test_passages: Sequence[str],
    *,
    n: int = 13,
) -> None:
    leaks = find_leaks(train_val_passages, test_passages, n=n)
    if leaks:
        sample = leaks[0][2]
        raise LeakageError(
            f"found {len(leaks)} test passage(s) sharing a {n}-gram with train/val; "
            f"example: {sample!r}"
        )


def assert_items_no_leakage(items: Iterable[Item], *, n: int = 13) -> None:
    train_val = [
        item.passage
        for item in items
        if item.split in {"train", "val"} and normalize_text(item.passage)
    ]
    test = [item.passage for item in items if item.split == "test" and normalize_text(item.passage)]
    assert_no_leakage(train_val, test, n=n)
