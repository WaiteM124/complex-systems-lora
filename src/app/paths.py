"""Locate the repository root so config and prompts stay out of the package."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("could not locate repository root (no pyproject.toml)")
