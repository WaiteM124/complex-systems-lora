"""Route items by licence tier and assert the repo stays clean."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from app.data.sources import load_sources
from app.paths import repo_root
from app.schema import Item

AWS_DOC_EXAMPLE_ACCOUNT = "123456789012"
ACCOUNT_ID_RE = re.compile(r"\b\d{12}\b")
SOURCE_ID_RE = re.compile(r"\bcs-\d{3}\b")

ALLOWED_SOURCE_ID_PATH_PREFIXES = (
    "config/sources.yaml",
    "docs/",
    "tests/",
    "CHANGELOG.md",
    "README.md",
    "AGENTS.md",
    "evals/",
)


def _tier_b_or_c_ids() -> frozenset[str]:
    return frozenset(
        source.id for source in load_sources().sources if source.licence_class in {"B", "C"}
    )


def route_item(item: Item, *, private_root: Path, public_root: Path) -> Path:
    """Return the destination path for an item (does not write)."""
    if item.licence_class in {"B", "N"}:
        return private_root / f"{item.licence_class.lower()}_{item.split}.jsonl"
    if item.split == "test":
        return public_root / "evals" / "datasets" / "tier_a_test.jsonl"
    return public_root / "data" / "public" / f"{item.split}.jsonl"


def write_routed_items(
    items: Sequence[Item],
    *,
    private_root: Path | None = None,
    repo: Path | None = None,
) -> dict[str, int]:
    root = repo or repo_root()
    private = private_root or (root / "data" / "private")
    private.mkdir(parents=True, exist_ok=True)
    (root / "data" / "public").mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    buckets: dict[Path, list[Item]] = {}
    for item in items:
        path = route_item(item, private_root=private, public_root=root)
        buckets.setdefault(path, []).append(item)
    for path, bucket in buckets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in bucket:
                handle.write(item.model_dump_json() + "\n")
        counts[str(path)] = len(bucket)
    return counts


def _git_ls_files(repo: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [
            str(path.relative_to(repo))
            for path in repo.rglob("*")
            if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts
        ]
    return [line for line in result.stdout.splitlines() if line.strip()]


def assert_repo_clean(paths: Sequence[str] | None = None, *, repo: Path | None = None) -> None:
    """Fail if private tiers, private paths, or account IDs leak into tracked files."""
    root = repo or repo_root()
    tracked = list(paths) if paths is not None else _git_ls_files(root)
    blocked_ids = _tier_b_or_c_ids()
    errors: list[str] = []

    for rel in tracked:
        if rel.startswith("data/private/") or rel == "data/private":
            errors.append(f"tracked private path: {rel}")

    for rel in tracked:
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if rel.endswith(".jsonl") and (rel.startswith("evals/") or rel.startswith("data/")):
            for line_no, raw in enumerate(text.splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                licence = obj.get("licence_class")
                if licence in {"B", "N", "C"}:
                    errors.append(f"{rel}:{line_no} committed JSONL has licence_class={licence}")

        allow = any(
            rel == prefix.rstrip("/") or rel.startswith(prefix)
            for prefix in ALLOWED_SOURCE_ID_PATH_PREFIXES
        )
        if not allow:
            for match in SOURCE_ID_RE.findall(text):
                if match in blocked_ids:
                    errors.append(f"{rel} contains tier-B/C source_id {match}")

        for match in ACCOUNT_ID_RE.findall(text):
            if match != AWS_DOC_EXAMPLE_ACCOUNT:
                errors.append(f"{rel} contains 12-digit number {match}")

    if errors:
        raise AssertionError("repo cleanliness failed:\n" + "\n".join(errors[:50]))
