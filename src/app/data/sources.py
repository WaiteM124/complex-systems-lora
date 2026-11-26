"""Load and validate config/sources.yaml. Refuse tier C for ingest."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from app.paths import repo_root

LicenceClass = Literal["A", "B", "C"]


class Source(BaseModel):
    id: str
    module: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str | None = None
    licence: str | None = None
    licence_url: str | None = None
    licence_class: LicenceClass
    nc_flag: bool = False
    nd_flag: bool = False
    attribution_text: str | None = None
    format: str
    ingest: bool = False

    @model_validator(mode="after")
    def ingest_rules(self) -> Source:
        if self.ingest and self.licence_class == "C":
            raise ValueError(f"tier-C source {self.id} cannot be ingested")
        return self


class SourcesFile(BaseModel):
    version: int = 1
    collection: str
    sources: list[Source]

    @field_validator("sources")
    @classmethod
    def require_forty_six(cls, value: list[Source]) -> list[Source]:
        ids = [source.id for source in value]
        if len(ids) != 46:
            raise ValueError(f"expected 46 sources, got {len(ids)}")
        if len(set(ids)) != 46:
            raise ValueError("source ids must be unique")
        return value


class TierCError(ValueError):
    """Raised when a tier-C source is requested for ingest."""


@lru_cache(maxsize=1)
def load_sources(path: Path | None = None) -> SourcesFile:
    sources_path = path or (repo_root() / "config" / "sources.yaml")
    raw: Any = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    return SourcesFile.model_validate(raw)


def clear_sources_cache() -> None:
    load_sources.cache_clear()


def get_source(source_id: str, *, path: Path | None = None) -> Source:
    for source in load_sources(path).sources:
        if source.id == source_id:
            return source
    raise KeyError(f"unknown source {source_id!r}")


def ingestable_sources(*, path: Path | None = None) -> list[Source]:
    return [source for source in load_sources(path).sources if source.ingest]


def ingestable_ids(*, path: Path | None = None) -> frozenset[str]:
    return frozenset(source.id for source in ingestable_sources(path=path))


def require_ingestable(source_id: str, *, path: Path | None = None) -> Source:
    source = get_source(source_id, path=path)
    if source.licence_class == "C":
        raise TierCError(f"tier-C source {source_id} is refused")
    if not source.ingest:
        raise TierCError(f"source {source_id} is not marked ingest=true")
    return source
