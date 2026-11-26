"""CLI entry: `python -m app <subcommand>`."""

from __future__ import annotations

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
