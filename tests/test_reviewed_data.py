from __future__ import annotations

from pathlib import Path

import pytest

from _synthetic import make_item, make_items, write_items
from app.data.reviewed import (
    DatasetNotFound,
    SplitLeakError,
    assert_no_test_items,
    find_reviewed_jsonl,
    items_sha256,
    load_items,
    split_items,
)


def test_override_file_and_dir(tmp_path: Path) -> None:
    path = write_items(tmp_path / "x" / "reviewed.jsonl", make_items())
    assert find_reviewed_jsonl(path) == path
    assert find_reviewed_jsonl(path.parent) == path
    with pytest.raises(DatasetNotFound):
        find_reviewed_jsonl(tmp_path / "missing.jsonl")


def test_glob_finds_nested_kaggle_mount(tmp_path: Path) -> None:
    root = tmp_path / "input"
    path = write_items(root / "datasets" / "owner" / "slug" / "reviewed.jsonl", make_items())
    assert find_reviewed_jsonl(search_roots=[root]) == path.resolve()


def test_glob_refuses_none_or_many(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFound):
        find_reviewed_jsonl(search_roots=[tmp_path])
    write_items(tmp_path / "a" / "reviewed.jsonl", make_items())
    write_items(tmp_path / "b" / "reviewed.jsonl", make_items())
    with pytest.raises(DatasetNotFound, match="more than one"):
        find_reviewed_jsonl(search_roots=[tmp_path])


def test_load_and_split(tmp_path: Path) -> None:
    path = write_items(tmp_path / "reviewed.jsonl", make_items(6, 3, 2))
    splits = split_items(load_items(path))
    assert splits.counts() == {"train": 6, "val": 3, "test": 2}


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    item = make_item(0, "train")
    path = write_items(tmp_path / "reviewed.jsonl", [item, item])
    with pytest.raises(ValueError, match="duplicate"):
        load_items(path)


def test_no_test_items_in_training() -> None:
    splits = split_items(make_items())
    assert_no_test_items([*splits.train, *splits.val], splits.test)
    with pytest.raises(SplitLeakError):
        assert_no_test_items([*splits.train, splits.test[0]], splits.test)
    copied = splits.test[0].model_copy(update={"split": "train", "id": "other"})
    with pytest.raises(SplitLeakError):
        assert_no_test_items([*splits.train, copied], splits.test)


def test_items_sha_is_order_independent() -> None:
    test = split_items(make_items()).test
    assert items_sha256(test) == items_sha256(list(reversed(test)))
    assert items_sha256(test) != items_sha256(test[:-1])
