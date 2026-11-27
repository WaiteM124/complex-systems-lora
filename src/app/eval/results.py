"""Aggregate results files (`results/<run>.json`). Numbers and ids only, never text.

The models forbid extra fields so that a results file cannot carry passages,
questions or model outputs by accident.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.eval.predictions import Judgment, Prediction
from app.schema import Item
from app.scoring import ScoredExample, score_examples

SCHEMA_VERSION = 1
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

Metric = float | int | None


class JudgeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    rubric_version: int
    mean: float | None
    n: int
    n_unparsed: int


class SystemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    adapter: str | None = None
    hardware: str
    decoding: str = "greedy"
    metrics: dict[str, Metric]
    judge: JudgeSummary | None = None


class RunResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run: str
    created_at: str
    updated_at: str
    dataset_sha256: str
    test_split_sha256: str
    n_test_items: int
    answer_prompt_version: int
    systems: dict[str, SystemResult] = Field(default_factory=dict)

    @field_validator("run")
    @classmethod
    def valid_run(cls, value: str) -> str:
        if not RUN_NAME_RE.match(value):
            raise ValueError("run name must be 1-64 chars of [A-Za-z0-9._-]")
        return value

    @field_validator("dataset_sha256", "test_split_sha256")
    @classmethod
    def valid_sha(cls, value: str) -> str:
        if not SHA256_RE.match(value):
            raise ValueError("expected a lowercase hex SHA-256")
        return value


class CoverageError(ValueError):
    """Predictions do not cover the test split exactly."""


def score_system(
    test_items: Sequence[Item],
    predictions: Sequence[Prediction],
    *,
    allow_partial: bool = False,
) -> dict[str, Metric]:
    by_id = {p.id: p for p in predictions}
    test_ids = {item.id for item in test_items}
    unknown = set(by_id) - test_ids
    if unknown:
        raise CoverageError(f"{len(unknown)} predictions are not test-split items")
    missing = test_ids - set(by_id)
    if missing and not allow_partial:
        raise CoverageError(f"{len(missing)} test items have no prediction")
    examples = [
        ScoredExample(
            gold_answerable=item.target.answerable,
            gold_concepts=tuple(item.target.concepts),
            passage=item.passage,
            output=by_id[item.id].output,
            latency_ms=by_id[item.id].latency_ms,
            output_tokens=by_id[item.id].output_tokens,
        )
        for item in sorted(test_items, key=lambda i: i.id)
        if item.id in by_id
    ]
    metrics = score_examples(examples)
    metrics["coverage"] = round(len(examples) / len(test_items), 4) if test_items else None
    return metrics


def summarise_judgments(judgments: Sequence[Judgment], test_ids: set[str]) -> JudgeSummary | None:
    relevant = [j for j in judgments if j.id in test_ids]
    if not relevant:
        return None
    scored = [j.score for j in relevant if j.score is not None]
    return JudgeSummary(
        model=relevant[0].judge_model,
        rubric_version=relevant[0].rubric_version,
        mean=round(sum(scored) / len(scored), 3) if scored else None,
        n=len(scored),
        n_unparsed=len(relevant) - len(scored),
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_results(path: Path) -> RunResults:
    return RunResults.model_validate_json(path.read_text(encoding="utf-8"))


def write_results(path: Path, results: RunResults) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.model_dump(mode="json")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_system(
    existing: RunResults | None,
    *,
    run: str,
    dataset_sha256: str,
    test_split_sha256: str,
    n_test_items: int,
    answer_prompt_version: int,
    system: str,
    result: SystemResult,
) -> RunResults:
    """Add or replace one system in a run; refuse to mix different test splits."""
    stamp = now_iso()
    if existing is None:
        existing = RunResults(
            run=run,
            created_at=stamp,
            updated_at=stamp,
            dataset_sha256=dataset_sha256,
            test_split_sha256=test_split_sha256,
            n_test_items=n_test_items,
            answer_prompt_version=answer_prompt_version,
        )
    if existing.test_split_sha256 != test_split_sha256:
        raise ValueError(f"run {run!r} was scored on a different test split; use a new run name")
    systems = dict(existing.systems)
    systems[system] = result
    return existing.model_copy(update={"systems": systems, "updated_at": stamp})
