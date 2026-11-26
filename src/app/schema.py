"""Grounded-QA item schema for the complex-systems data pipeline."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from app.paths import repo_root

LicenceClass = Literal["A", "B", "N"]
SplitName = Literal["train", "val", "test"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [p for p in _SENTENCE_SPLIT.split(stripped) if p.strip()]
    return len(parts) if parts else 1


@lru_cache(maxsize=1)
def load_concept_vocabulary(path: Path | None = None) -> frozenset[str]:
    concepts_path = path or (repo_root() / "config" / "concepts.yaml")
    raw: Any = yaml.safe_load(concepts_path.read_text(encoding="utf-8"))
    tags: set[str] = set()
    modules = raw.get("modules", {})
    for values in modules.values():
        tags.update(str(tag) for tag in values)
    return frozenset(tags)


class Target(BaseModel):
    answerable: bool
    answer: str = ""
    quote: str = ""
    concepts: list[str] = Field(default_factory=list)

    @field_validator("concepts")
    @classmethod
    def validate_concepts(cls, value: list[str]) -> list[str]:
        vocabulary = load_concept_vocabulary()
        unknown = [tag for tag in value if tag not in vocabulary]
        if unknown:
            raise ValueError(f"concepts not in vocabulary: {unknown}")
        if len(value) > 3:
            raise ValueError("concepts must hold at most 3 tags")
        return value

    @model_validator(mode="after")
    def validate_target_rules(self) -> Target:
        if self.answerable:
            if not self.answer.strip():
                raise ValueError("answerable items require a non-empty answer")
            if count_sentences(self.answer) > 3:
                raise ValueError("answer must be at most 3 sentences")
            if not self.quote.strip():
                raise ValueError("answerable items require a non-empty quote")
            if count_sentences(self.quote) > 2:
                raise ValueError("quote must be at most 2 sentences")
            if not (1 <= len(self.concepts) <= 3):
                raise ValueError("answerable items require 1 to 3 concepts")
        else:
            if self.answer.strip() or self.quote.strip() or self.concepts:
                raise ValueError("unanswerable items require empty answer, quote, and concepts")
        return self


class Item(BaseModel):
    id: str
    source_id: str
    licence_class: LicenceClass
    split: SplitName
    section_id: str
    passage: str
    question: str
    target: Target
    source: str | None = None

    @field_validator("licence_class", mode="before")
    @classmethod
    def reject_licence_c(cls, value: object) -> object:
        if value == "C":
            raise ValueError("licence_class C is rejected")
        return value

    @model_validator(mode="after")
    def quote_must_be_substring(self) -> Item:
        quote = self.target.quote
        if quote and quote not in self.passage:
            raise ValueError("quote must be a verbatim substring of passage")
        return self
