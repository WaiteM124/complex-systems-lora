"""Load versioned prompt files from the prompts/ directory."""

from __future__ import annotations

from pathlib import Path

from app.paths import repo_root


def load_prompt(name: str, version: int, *, directory: Path | None = None) -> str:
    """Return the text of `prompts/{name}.v{version}.txt`."""
    root = directory or (repo_root() / "prompts")
    path = root / f"{name}.v{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")
