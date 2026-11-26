"""Load model ids and generation parameters from config/models.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.paths import repo_root


class ModelParams(BaseModel):
    id: str
    temperature: float = 0.2
    max_tokens: int = Field(gt=0)


class ModelsFile(BaseModel):
    default_environment: str
    environments: dict[str, dict[str, ModelParams]]


@lru_cache(maxsize=1)
def load_models(path: Path | None = None) -> ModelsFile:
    models_path = path or (repo_root() / "config" / "models.yaml")
    raw: Any = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    return ModelsFile.model_validate(raw)


def get_model(
    name: str = "chat",
    environment: str | None = None,
    *,
    path: Path | None = None,
) -> ModelParams:
    config = load_models(path)
    env = environment or config.default_environment
    try:
        models = config.environments[env]
    except KeyError as exc:
        raise KeyError(f"unknown model environment {env!r}") from exc
    try:
        return models[name]
    except KeyError as exc:
        raise KeyError(f"unknown model {name!r} in environment {env!r}") from exc
