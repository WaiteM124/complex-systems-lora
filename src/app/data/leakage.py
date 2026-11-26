"""Fail if any test passage shares a 13-gram with train or validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.data.filters import ngrams, normalize_text
from app.schema import Item


class LeakageError(ValueError):
    """Raised when train/val and test passages leak n-grams."""


def passage_ngrams(passage: str, n: int = 13) -> set[str]:
    return ngrams(passage, n)


def _build_train_val_index(items: Iterable[Item], n: int = 13) -> dict[str, str]:
    """Build an index of n-grams from train/val passages."""
    index: dict[str, str] = {}
    for item in items:
        if item.split in {"train", "val"} and normalize_text(item.passage):
            for gram in passage_ngrams(item.passage, n):
                if gram and gram not in index:
                    index[gram] = item.passage
    return index


def leaking_test_items(items: Iterable[Item], n: int = 13) -> list[Item]:
    """Return test-split Items whose passage shares any n-gram with train/val."""
    items_list = list(items)
    index = _build_train_val_index(items_list, n)
    leaking: list[Item] = []
    for item in items_list:
        if item.split != "test" or not normalize_text(item.passage):
            continue
        for gram in passage_ngrams(item.passage, n):
            if gram in index:
                leaking.append(item)
                break
    return leaking


def drop_leaking_test_items(items: Iterable[Item], n: int = 13) -> tuple[list[Item], list[Item]]:
    """Drop test items that leak n-grams with train/val.

    Returns (kept, dropped) where kept has all train/val items plus non-leaking
    test items, and dropped has the leaking test items.
    """
    items_list = list(items)
    leaking_set = set(id(item) for item in leaking_test_items(items_list, n))
    kept: list[Item] = []
    dropped: list[Item] = []
    for item in items_list:
        if id(item) in leaking_set:
            dropped.append(item)
        else:
            kept.append(item)
    return kept, dropped


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
