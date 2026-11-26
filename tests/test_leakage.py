from __future__ import annotations

from pathlib import Path

import pytest

from app.data.leakage import (
    LeakageError,
    assert_no_leakage,
    drop_leaking_test_items,
    find_leaks,
    leaking_test_items,
)
from app.schema import Item


def _make_item(
    item_id: str,
    split: str,
    passage: str,
    *,
    source_id: str = "cs-018",
    licence_class: str = "A",
) -> Item:
    """Create a fake Item for testing."""
    return Item(
        id=item_id,
        source_id=source_id,
        licence_class=licence_class,
        split=split,
        section_id="sec-001",
        passage=passage,
        question="What is discussed?",
        target={
            "answerable": True,
            "answer": "Something.",
            "quote": passage[:20] if len(passage) >= 20 else passage,
            "concepts": ["complex-system"],
        },
    )


def test_planted_13gram_caught() -> None:
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    train = [f"intro text {shared} trailing train words"]
    test = [f"other intro {shared} trailing test words"]
    leaks = find_leaks(train, test, n=13)
    assert leaks
    with pytest.raises(LeakageError):
        assert_no_leakage(train, test, n=13)


def test_no_leak_when_disjoint() -> None:
    train = ["completely different train passage about networks and hubs"]
    test = ["unrelated test passage discussing entropy and information"]
    assert find_leaks(train, test, n=13) == []
    assert_no_leakage(train, test, n=13)


class TestLeakingTestItems:
    """Tests for leaking_test_items function."""

    def test_identifies_leaking_test_item(self) -> None:
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_item = _make_item("train-1", "train", f"intro {shared} end")
        test_item = _make_item("test-1", "test", f"start {shared} outro")
        items = [train_item, test_item]
        leaking = leaking_test_items(items, n=13)
        assert len(leaking) == 1
        assert leaking[0].id == "test-1"

    def test_train_val_items_never_in_leaking(self) -> None:
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_item = _make_item("train-1", "train", f"intro {shared} end")
        val_item = _make_item("val-1", "val", f"start {shared} outro")
        items = [train_item, val_item]
        leaking = leaking_test_items(items, n=13)
        assert leaking == []

    def test_no_leaking_when_disjoint(self) -> None:
        train_item = _make_item("train-1", "train", "completely unique train passage content here")
        test_item = _make_item("test-1", "test", "unrelated test passage about different things")
        items = [train_item, test_item]
        leaking = leaking_test_items(items, n=13)
        assert leaking == []


class TestDropLeakingTestItems:
    """Tests for drop_leaking_test_items function."""

    def test_drops_leaking_test_keeps_train_val(self) -> None:
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_item = _make_item("train-1", "train", f"intro {shared} end")
        val_item = _make_item("val-1", "val", f"start {shared} here and more")
        test_item = _make_item("test-1", "test", f"begin {shared} outro")
        clean_test = _make_item("test-2", "test", "unrelated unique test content")
        items = [train_item, val_item, test_item, clean_test]

        kept, dropped = drop_leaking_test_items(items, n=13)

        assert len(dropped) == 1
        assert dropped[0].id == "test-1"
        assert len(kept) == 3
        kept_ids = {item.id for item in kept}
        assert kept_ids == {"train-1", "val-1", "test-2"}

    def test_clean_data_drops_nothing(self) -> None:
        train_item = _make_item("train-1", "train", "unique train passage with no overlap")
        val_item = _make_item("val-1", "val", "another unique validation passage here")
        test_item = _make_item("test-1", "test", "completely separate test content")
        items = [train_item, val_item, test_item]

        kept, dropped = drop_leaking_test_items(items, n=13)

        assert dropped == []
        assert len(kept) == 3

    def test_all_train_val_preserved(self) -> None:
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_items = [
            _make_item(f"train-{i}", "train", f"content {shared} number {i}") for i in range(3)
        ]
        val_item = _make_item("val-1", "val", f"validation {shared} content")
        test_item = _make_item("test-1", "test", f"test {shared} leaking")
        items = train_items + [val_item, test_item]

        kept, dropped = drop_leaking_test_items(items, n=13)

        kept_train = [item for item in kept if item.split == "train"]
        kept_val = [item for item in kept if item.split == "val"]
        assert len(kept_train) == 3
        assert len(kept_val) == 1
        assert len(dropped) == 1


class TestCLILeakage:
    """Tests for the leakage CLI command."""

    def test_drop_leaks_rewrites_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.cli import cmd_leakage

        private = tmp_path / "private"
        private.mkdir()
        filtered = private / "filtered.jsonl"

        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_item = _make_item("train-1", "train", f"intro {shared} end")
        test_leak = _make_item("test-1", "test", f"begin {shared} outro")
        test_clean = _make_item("test-2", "test", "unrelated unique test passage")

        with filtered.open("w") as f:
            for item in [train_item, test_leak, test_clean]:
                f.write(item.model_dump_json() + "\n")

        class FakeSettings:
            def __init__(self, _env_file: str | None = None) -> None:
                self.private_data_dir = private

        monkeypatch.setattr("app.cli.Settings", FakeSettings)
        monkeypatch.setattr("app.cli.repo_root", lambda: tmp_path)
        monkeypatch.setattr("app.data.licence.repo_root", lambda: tmp_path)

        class Args:
            no_env = True
            drop_leaks = True

        result = cmd_leakage(Args())
        assert result == 0

        reloaded = []
        for line in filtered.read_text().splitlines():
            if line.strip():
                reloaded.append(Item.model_validate_json(line))
        assert len(reloaded) == 2
        reloaded_ids = {item.id for item in reloaded}
        assert reloaded_ids == {"train-1", "test-2"}

        dropped_path = private / "leakage_dropped.jsonl"
        assert dropped_path.exists()
        dropped_items = []
        for line in dropped_path.read_text().splitlines():
            if line.strip():
                dropped_items.append(Item.model_validate_json(line))
        assert len(dropped_items) == 1
        assert dropped_items[0].id == "test-1"

    def test_no_flag_exits_1_with_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from app.cli import cmd_leakage

        private = tmp_path / "private"
        private.mkdir()
        filtered = private / "filtered.jsonl"

        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_item = _make_item("train-1", "train", f"intro {shared} end")
        test_leak = _make_item("test-1", "test", f"begin {shared} outro")

        with filtered.open("w") as f:
            for item in [train_item, test_leak]:
                f.write(item.model_dump_json() + "\n")

        class FakeSettings:
            def __init__(self, _env_file: str | None = None) -> None:
                self.private_data_dir = private

        monkeypatch.setattr("app.cli.Settings", FakeSettings)

        class Args:
            no_env = True
            drop_leaks = False

        result = cmd_leakage(Args())
        assert result == 1

        captured = capsys.readouterr()
        assert "--drop-leaks" in captured.err

    def test_clean_data_passes_without_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.cli import cmd_leakage

        private = tmp_path / "private"
        private.mkdir()
        filtered = private / "filtered.jsonl"

        train_item = _make_item("train-1", "train", "unique train content no overlap")
        test_item = _make_item("test-1", "test", "different unique test passage content")

        with filtered.open("w") as f:
            for item in [train_item, test_item]:
                f.write(item.model_dump_json() + "\n")

        class FakeSettings:
            def __init__(self, _env_file: str | None = None) -> None:
                self.private_data_dir = private

        monkeypatch.setattr("app.cli.Settings", FakeSettings)

        class Args:
            no_env = True
            drop_leaks = False

        result = cmd_leakage(Args())
        assert result == 0

    def test_drop_leaks_passes_on_rerun(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.cli import cmd_leakage

        private = tmp_path / "private"
        private.mkdir()
        filtered = private / "filtered.jsonl"

        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        train_item = _make_item("train-1", "train", f"intro {shared} end")
        test_leak = _make_item("test-1", "test", f"begin {shared} outro")
        test_clean = _make_item("test-2", "test", "unrelated unique test passage")

        with filtered.open("w") as f:
            for item in [train_item, test_leak, test_clean]:
                f.write(item.model_dump_json() + "\n")

        class FakeSettings:
            def __init__(self, _env_file: str | None = None) -> None:
                self.private_data_dir = private

        monkeypatch.setattr("app.cli.Settings", FakeSettings)
        monkeypatch.setattr("app.cli.repo_root", lambda: tmp_path)
        monkeypatch.setattr("app.data.licence.repo_root", lambda: tmp_path)

        class DropArgs:
            no_env = True
            drop_leaks = True

        result1 = cmd_leakage(DropArgs())
        assert result1 == 0

        class CheckArgs:
            no_env = True
            drop_leaks = False

        result2 = cmd_leakage(CheckArgs())
        assert result2 == 0
