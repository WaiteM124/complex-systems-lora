from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.paths import repo_root
from app.settings import Settings


def _example_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0])
    return keys


def test_settings_fail_closed_when_app_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_fail_closed_without_llm_key_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_load_when_app_env_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    settings = Settings(_env_file=None)
    assert settings.app_env == "local"


def test_env_example_matches_settings_fields() -> None:
    example_keys = _example_env_keys(repo_root() / ".env.example")
    field_keys = {name.upper() for name in Settings.model_fields}
    assert example_keys == field_keys
