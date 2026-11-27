"""Prediction and judgment records (private JSONL; may hold model text)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    system: str
    output: str
    model: str
    adapter: str | None = None
    hardware: str
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    system: str
    judge_model: str
    rubric_version: int
    score: int | None
    rationale: str = ""


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        return []
    out: list[T] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(model.model_validate_json(line))
        except ValueError:
            continue  # torn trailing line after an interrupted run
    return out


def append_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()


def done_ids(records: Iterable[Prediction | Judgment]) -> set[str]:
    return {record.id for record in records}


def predictions_path(directory: Path, system: str) -> Path:
    return directory / f"{system}.jsonl"
