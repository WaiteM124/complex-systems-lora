"""Locate, load and split the reviewed dataset (`reviewed.jsonl`).

Each line is an `app.schema.Item`; the split is the item's own `split` field
(`train`, `val` or `test`), set by the section-level split in milestone 1 and
kept by `import-review`. No passage text leaves this module except to callers
that train or generate privately.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.schema import Item

DATASET_FILENAME = "reviewed.jsonl"
DEFAULT_SEARCH_ROOTS: tuple[str, ...] = ("/kaggle/input",)


class DatasetNotFound(FileNotFoundError):
    """Raised when reviewed.jsonl cannot be located."""


class SplitLeakError(AssertionError):
    """Raised when a test item would be trained on."""


def find_reviewed_jsonl(
    override: str | Path | None = None,
    *,
    search_roots: Sequence[str | Path] = DEFAULT_SEARCH_ROOTS,
    filename: str = DATASET_FILENAME,
) -> Path:
    """Return the dataset path: explicit override first, else glob the search roots.

    Kaggle mounts datasets under `/kaggle/input/<slug>/` (or deeper, depending on
    how it was attached), so the roots are searched recursively for `filename`.
    More than one match is ambiguous and refused; pass an override instead.
    """
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise DatasetNotFound(f"dataset override does not exist: {path}")
        return path
    matches: list[Path] = []
    for root in search_roots:
        root_path = Path(root)
        if root_path.is_dir():
            matches.extend(sorted(p for p in root_path.glob(f"**/{filename}") if p.is_file()))
    unique = sorted({p.resolve() for p in matches})
    if not unique:
        roots = ", ".join(str(r) for r in search_roots)
        raise DatasetNotFound(
            f"no {filename} under {roots}; attach the private Kaggle dataset "
            "or pass --data /path/to/reviewed.jsonl"
        )
    if len(unique) > 1:
        listed = ", ".join(str(p) for p in unique)
        raise DatasetNotFound(f"more than one {filename} found ({listed}); pass --data")
    return unique[0]


def load_items(path: Path) -> list[Item]:
    items: list[Item] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            items.append(Item.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 — report the line, not the text
            raise ValueError(f"{path.name}:{line_no} is not a valid item") from exc
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} has duplicate item ids")
    return items


@dataclass(frozen=True)
class Splits:
    train: list[Item]
    val: list[Item]
    test: list[Item]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def split_items(items: Iterable[Item]) -> Splits:
    train: list[Item] = []
    val: list[Item] = []
    test: list[Item] = []
    for item in items:
        if item.split == "train":
            train.append(item)
        elif item.split == "val":
            val.append(item)
        else:
            test.append(item)
    return Splits(train=train, val=val, test=test)


def assert_no_test_items(trained_on: Sequence[Item], test: Sequence[Item]) -> None:
    """Fail if any training/validation item is a test item (by split, id or passage)."""
    bad_split = [item.id for item in trained_on if item.split == "test"]
    if bad_split:
        raise SplitLeakError(f"{len(bad_split)} test-split items in the training set")
    test_ids = {item.id for item in test}
    test_sections = {item.section_id for item in test}
    test_passages = {item.passage for item in test}
    overlap_ids = [item.id for item in trained_on if item.id in test_ids]
    overlap_sections = [item.id for item in trained_on if item.section_id in test_sections]
    overlap_passages = [item.id for item in trained_on if item.passage in test_passages]
    if overlap_ids or overlap_sections or overlap_passages:
        raise SplitLeakError(
            "training items overlap the test split: "
            f"ids={len(overlap_ids)} sections={len(overlap_sections)} "
            f"passages={len(overlap_passages)}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def items_sha256(items: Sequence[Item]) -> str:
    """Order-independent SHA-256 of a set of items (canonical JSON, sorted by id)."""
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.id):
        digest.update(json.dumps(item.model_dump(), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
