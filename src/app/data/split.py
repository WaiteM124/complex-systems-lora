"""Deterministic train/val/test split by section ID, before generation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.data.chunk import Chunk, load_data_config
from app.paths import repo_root

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitAssignment:
    section_id: str
    split: SplitName


def _ratios(config: dict[str, Any] | None = None) -> tuple[float, float, float, int]:
    cfg = config or load_data_config()
    split_cfg = cfg.get("split", {})
    train = float(split_cfg.get("train", 0.8))
    val = float(split_cfg.get("val", 0.1))
    test = float(split_cfg.get("test", 0.1))
    seed = int(split_cfg.get("seed", 20261126))
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    return train, val, test, seed


def assign_sections(
    section_ids: Sequence[str],
    *,
    seed: int | None = None,
    ratios: tuple[float, float, float] | None = None,
) -> dict[str, SplitName]:
    unique = sorted(set(section_ids))
    if not unique:
        return {}
    config = load_data_config()
    train_r, val_r, test_r, default_seed = _ratios(config)
    if ratios is not None:
        train_r, val_r, test_r = ratios
    use_seed = default_seed if seed is None else seed
    rng = random.Random(use_seed)
    order = unique[:]
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, round(n * test_r)) if n >= 3 else (1 if n >= 1 else 0)
    n_val = max(1, round(n * val_r)) if n >= 3 else (1 if n >= 2 else 0)
    # Ensure train gets the remainder and totals match.
    while n_test + n_val >= n and n_val > 0:
        n_val -= 1
    while n_test + n_val >= n and n_test > 0:
        n_test -= 1
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test : n_test + n_val])
    assignment: dict[str, SplitName] = {}
    for section_id in unique:
        if section_id in test_ids:
            assignment[section_id] = "test"
        elif section_id in val_ids:
            assignment[section_id] = "val"
        else:
            assignment[section_id] = "train"
    return assignment


def split_chunks(
    chunks: Iterable[Chunk],
    *,
    seed: int | None = None,
) -> dict[SplitName, list[Chunk]]:
    chunk_list = list(chunks)
    section_ids = [chunk.section_id for chunk in chunk_list]
    assignment = assign_sections(section_ids, seed=seed)
    out: dict[SplitName, list[Chunk]] = {"train": [], "val": [], "test": []}
    for chunk in chunk_list:
        out[assignment[chunk.section_id]].append(chunk)
    return out


def write_test_sections(
    assignment: dict[str, SplitName],
    path: Path | None = None,
) -> Path:
    out_path = path or (repo_root() / "data" / "splits" / "test_sections.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_ids = sorted(sid for sid, split in assignment.items() if split == "test")
    out_path.write_text("\n".join(test_ids) + ("\n" if test_ids else ""), encoding="utf-8")
    return out_path


def read_test_sections(path: Path | None = None) -> list[str]:
    split_path = path or (repo_root() / "data" / "splits" / "test_sections.txt")
    if not split_path.exists():
        return []
    return [
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_assignment_json(
    assignment: dict[str, SplitName],
    path: Path | None = None,
) -> Path:
    out_path = path or (repo_root() / "data" / "splits" / "section_assignment.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_split_seed() -> int:
    _, _, _, seed = _ratios()
    return seed
