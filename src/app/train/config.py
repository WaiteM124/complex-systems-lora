"""Typed loader for config/train.yaml. Pure; no GPU imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.paths import repo_root


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    path: str = ""
    search_roots: list[str] = Field(default_factory=lambda: ["/kaggle/input"])
    filename: str = "reviewed.jsonl"


class PromptConfig(_Strict):
    answer_system_version: int = 1
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"


class ModelLoadConfig(_Strict):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    max_seq_length: int = Field(default=2048, gt=0)


class LoraConfig(_Strict):
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = 0.0
    target_modules: list[str]
    bias: str = "none"


class TrainingConfig(_Strict):
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 3
    per_device_train_batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 5
    save_steps: int = Field(default=25, gt=0)
    save_total_limit: int | None = 12
    seed: int = 20261126
    smoke_steps: int = Field(default=20, gt=0)


class FallbackRung(_Strict):
    per_device_train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    max_seq_length: int | None = None


class SelectionConfig(_Strict):
    val_items: int = Field(default=48, gt=0)
    max_new_tokens: int = Field(default=256, gt=0)
    max_candidates: int = Field(default=4, gt=0)


class OutputConfig(_Strict):
    working_dir: str = "/kaggle/working"
    checkpoints_subdir: str = "checkpoints"
    adapter_subdir: str = "adapter"
    summary_file: str = "train_summary.json"
    state_file: str = "train_state.json"

    def working(self) -> Path:
        return Path(self.working_dir)

    def checkpoints(self) -> Path:
        return self.working() / self.checkpoints_subdir

    def adapter(self) -> Path:
        return self.working() / self.adapter_subdir

    def summary(self) -> Path:
        return self.working() / self.summary_file

    def state(self) -> Path:
        return self.working() / self.state_file


class HubConfig(_Strict):
    push: bool = True
    repo_id: str
    private: bool = False


class TrainConfig(_Strict):
    data: DataConfig = Field(default_factory=DataConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    model: ModelLoadConfig = Field(default_factory=ModelLoadConfig)
    lora: LoraConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    oom_fallback: list[FallbackRung] = Field(default_factory=list)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hub: HubConfig


class Attempt(_Strict):
    """One rung of the OOM ladder: concrete batch, accumulation and sequence length."""

    rung: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int

    @property
    def effective_batch(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def default_config_path() -> Path:
    return repo_root() / "config" / "train.yaml"


def load_train_config(path: Path | None = None) -> TrainConfig:
    raw: Any = yaml.safe_load((path or default_config_path()).read_text(encoding="utf-8"))
    return TrainConfig.model_validate(raw)


def attempt_ladder(config: TrainConfig) -> list[Attempt]:
    """The recipe followed by each OOM fallback; later rungs inherit earlier values."""
    current = Attempt(
        rung=0,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_seq_length=config.model.max_seq_length,
    )
    ladder = [current]
    for index, rung in enumerate(config.oom_fallback, start=1):
        current = Attempt(
            rung=index,
            per_device_train_batch_size=(
                rung.per_device_train_batch_size or current.per_device_train_batch_size
            ),
            gradient_accumulation_steps=(
                rung.gradient_accumulation_steps or current.gradient_accumulation_steps
            ),
            max_seq_length=rung.max_seq_length or current.max_seq_length,
        )
        ladder.append(current)
    return ladder
