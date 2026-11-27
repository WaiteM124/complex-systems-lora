"""Typed loader for config/eval.yaml, plus dataset path resolution. Pure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.data.reviewed import find_reviewed_jsonl
from app.paths import repo_root


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalDataConfig(_Strict):
    path: str = ""
    search_roots: list[str] = Field(default_factory=lambda: ["/kaggle/input"])
    filename: str = "reviewed.jsonl"
    split: str = "test"


class EvalPromptConfig(_Strict):
    answer_system_version: int = 1
    judge_rubric_version: int = 1


class LocalConfig(_Strict):
    dtype: str = "float16"
    load_in_4bit: bool = False
    max_new_tokens: int = Field(default=384, gt=0)
    hardware_label_default: str = "unknown GPU"


class AdapterConfig(_Strict):
    local_dir: str = "/kaggle/working/adapter"
    search_roots: list[str] = Field(default_factory=lambda: ["/kaggle/input"])
    hub_repo: str


class BedrockEvalConfig(_Strict):
    answer_max_tokens: int = Field(default=400, gt=0)
    temperature: float = 0.0
    system_name: str = "qwen3-235b"
    hardware_label: str = "Bedrock eu-west-2"


class JudgeConfig(_Strict):
    systems: list[str] = Field(default_factory=lambda: ["base", "lora", "qwen3-235b"])


class EvalPaths(_Strict):
    kaggle_predictions_dir: str = "/kaggle/working/predictions"
    kaggle_results_dir: str = "/kaggle/working/results"
    local_predictions_dir: str = "data/private/predictions"
    local_judgments_dir: str = "data/private/judgments"
    results_dir: str = "results"

    @staticmethod
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root() / path


class EvalConfig(_Strict):
    data: EvalDataConfig = Field(default_factory=EvalDataConfig)
    prompt: EvalPromptConfig = Field(default_factory=EvalPromptConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)
    adapter: AdapterConfig
    bedrock: BedrockEvalConfig = Field(default_factory=BedrockEvalConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    paths: EvalPaths = Field(default_factory=EvalPaths)


def load_eval_config(path: Path | None = None) -> EvalConfig:
    raw: Any = yaml.safe_load(
        (path or repo_root() / "config" / "eval.yaml").read_text(encoding="utf-8")
    )
    return EvalConfig.model_validate(raw)


def resolve_data_path(
    cli_path: str | None,
    config: EvalConfig,
    *,
    private_dir: Path | None = None,
) -> Path:
    """CLI flag, then config path, then `<private_dir>/reviewed.jsonl`, then Kaggle glob."""
    explicit = cli_path or config.data.path or None
    if explicit:
        return find_reviewed_jsonl(explicit, filename=config.data.filename)
    if private_dir is not None:
        local = private_dir / config.data.filename
        if local.is_file():
            return local
    return find_reviewed_jsonl(
        None, search_roots=config.data.search_roots, filename=config.data.filename
    )
