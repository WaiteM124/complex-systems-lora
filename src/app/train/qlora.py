"""QLoRA fine-tune of the base model on one free T4.

Run: `python -m app.train.qlora --config config/train.yaml [--smoke]`

- Base model id comes from `config/models.yaml` (`base_model`).
- Unsloth + TRL `SFTTrainer` + PEFT; QLoRA nf4; loss on the assistant JSON only.
- `--smoke` runs `training.smoke_steps` steps and logs peak VRAM; it records the
  first rung of the OOM ladder that fits so the full run starts there.
- Full runs save resumable checkpoints under `<working>/checkpoints/rung-<n>/`,
  score the newest checkpoints by validation answerability F1, and save the best
  adapter to `<working>/adapter`.
- Pushes the adapter to the HF Hub only when `HF_TOKEN` is set (env var or Kaggle
  Secret). The token is never printed or logged.

GPU libraries (torch, unsloth, trl, peft, datasets) are imported lazily inside
functions so CI and tests stay CPU-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data.reviewed import (
    Splits,
    assert_no_test_items,
    file_sha256,
    find_reviewed_jsonl,
    items_sha256,
    load_items,
    split_items,
)
from app.model_config import get_base_model
from app.schema import Item
from app.scoring import ScoredExample, score_examples
from app.task import build_messages, build_system_prompt
from app.train.config import Attempt, TrainConfig, attempt_ladder, load_train_config

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
FINAL_NAME = "final"


# --------------------------------------------------------------------------- pure helpers


def log(event: str, **fields: Any) -> None:
    """One JSON line to stdout. Never pass secrets or item text here."""
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def select_val_subset(val: Sequence[Item], n: int, seed: int) -> list[Item]:
    ordered = sorted(val, key=lambda item: item.id)
    if len(ordered) <= n:
        return ordered
    return sorted(random.Random(seed).sample(ordered, n), key=lambda item: item.id)


@dataclass(frozen=True)
class PreparedData:
    splits: Splits
    train: list[Item]
    val_subset: list[Item]
    dataset_sha256: str
    test_split_sha256: str


def prepare_data(path: Path, config: TrainConfig) -> PreparedData:
    items = load_items(path)
    splits = split_items(items)
    if not splits.train:
        raise ValueError("no train-split items in the dataset")
    assert_no_test_items([*splits.train, *splits.val], splits.test)
    train = [item for item in splits.train if item.split == "train"]
    val_subset = select_val_subset(splits.val, config.selection.val_items, config.training.seed)
    return PreparedData(
        splits=splits,
        train=train,
        val_subset=val_subset,
        dataset_sha256=file_sha256(path),
        test_split_sha256=items_sha256(splits.test),
    )


def chat_records(items: Sequence[Item], system: str) -> list[list[dict[str, str]]]:
    """Chat messages (system, user, assistant) for supervised fine-tuning."""
    return [build_messages(item, system=system, include_target=True) for item in items]


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in text


def checkpoint_step(name: str) -> int | None:
    match = CHECKPOINT_RE.match(name)
    return int(match.group(1)) if match else None


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoint dirs sorted by step (oldest first)."""
    if not run_dir.is_dir():
        return []
    found = [
        (step, path)
        for path in run_dir.iterdir()
        if path.is_dir() and (step := checkpoint_step(path.name)) is not None
    ]
    return [path for _, path in sorted(found)]


def selection_candidates(run_dir: Path, max_candidates: int) -> list[Path]:
    """Newest checkpoints plus the final save, if present."""
    candidates = list_checkpoints(run_dir)[-max_candidates:]
    final = run_dir / FINAL_NAME
    if (final / "adapter_model.safetensors").is_file():
        candidates.append(final)
    return candidates


def pick_best(scores: dict[str, float | None]) -> str:
    """Highest score wins; ties go to the earlier candidate (less training)."""
    if not scores:
        raise ValueError("no candidates scored")
    best_name = ""
    best_score = -math.inf
    for name, score in scores.items():
        value = -math.inf if score is None else score
        if value > best_score:
            best_name, best_score = name, value
    return best_name or next(iter(scores))


def warmup_steps(n_examples: int, attempt: Attempt, epochs: float, ratio: float) -> int:
    steps_per_epoch = math.ceil(n_examples / attempt.effective_batch)
    total = math.ceil(steps_per_epoch * epochs)
    return max(0, math.ceil(total * ratio))


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def start_rung(state: dict[str, Any], forced: int | None, ladder_len: int) -> int:
    rung = forced if forced is not None else int(state.get("rung", 0))
    if not 0 <= rung < ladder_len:
        raise ValueError(f"rung {rung} outside the OOM ladder (0..{ladder_len - 1})")
    return rung


def hf_token() -> str | None:
    """HF token from the env or a Kaggle Secret. Never print the return value."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient  # Kaggle runtime only

        value = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 — absent secret or not on Kaggle
        return None
    if not value:
        return None
    return str(value).strip() or None


# --------------------------------------------------------------------------- GPU section


@dataclass
class AttemptResult:
    attempt: Attempt
    ok: bool
    oom: bool = False
    runtime_s: float = 0.0
    peak_vram_gib: float | None = None
    peak_alloc_gib: float | None = None
    train_loss: float | None = None
    global_step: int = 0
    dropped_too_long: int = 0


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return str(torch.cuda.get_device_name(0))


def _load_model(config: TrainConfig, attempt: Attempt, base_model: str) -> tuple[Any, Any]:
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=attempt.max_seq_length,
        dtype=torch.float16 if config.training.fp16 else None,
        load_in_4bit=config.model.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth" if config.training.gradient_checkpointing else False,
        random_state=config.training.seed,
    )
    return model, tokenizer


def _texts(
    tokenizer: Any, items: Sequence[Item], system: str, max_len: int
) -> tuple[list[str], int]:
    texts: list[str] = []
    dropped = 0
    for messages in chat_records(items, system):
        text = str(tokenizer.apply_chat_template(messages, tokenize=False))
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_len:
            dropped += 1
            continue
        texts.append(text)
    return texts, dropped


def _build_trainer(
    config: TrainConfig,
    attempt: Attempt,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    output_dir: Path,
    smoke: bool,
) -> Any:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    t = config.training
    args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=attempt.max_seq_length,
        packing=False,
        per_device_train_batch_size=attempt.per_device_train_batch_size,
        gradient_accumulation_steps=attempt.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.smoke_steps if smoke else -1,
        warmup_steps=warmup_steps(len(texts), attempt, t.num_train_epochs, t.warmup_ratio),
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,
        optim=t.optim,
        fp16=t.fp16,
        bf16=t.bf16,
        logging_steps=t.logging_steps,
        save_strategy="no" if smoke else "steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        report_to="none",
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        args=args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=config.prompt.instruction_part,
        response_part=config.prompt.response_part,
    )
    _check_masking(trainer)
    return trainer


def _check_masking(trainer: Any) -> None:
    """Fail early if the loss mask would train on the prompt or on nothing."""
    sample = trainer.train_dataset[0]
    labels = [int(x) for x in sample["labels"]]
    masked = sum(1 for x in labels if x == -100)
    if masked == 0 or masked == len(labels):
        raise RuntimeError(
            "assistant-only loss mask looks wrong "
            f"(masked {masked} of {len(labels)} tokens); check prompt markers in train.yaml"
        )


def _run_attempt(
    config: TrainConfig,
    attempt: Attempt,
    data: PreparedData,
    system: str,
    base_model: str,
    *,
    smoke: bool,
    previous: dict[str, Any] | None = None,
) -> tuple[AttemptResult, Any, Any]:
    import torch

    output = config.output
    run_dir = output.working() / "smoke" if smoke else output.checkpoints() / f"rung-{attempt.rung}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model: Any = None
    tokenizer: Any = None
    trainer: Any = None
    started = time.perf_counter()
    try:
        model, tokenizer = _load_model(config, attempt, base_model)
        final_done = (run_dir / FINAL_NAME / "adapter_model.safetensors").is_file()
        if not smoke and final_done and previous:
            # Training finished in an earlier session: go straight to selection.
            log("training_already_complete", run_dir=str(run_dir))
            prior = AttemptResult(attempt=attempt, ok=True, **previous)
            return prior, model, tokenizer
        texts, dropped = _texts(tokenizer, data.train, system, attempt.max_seq_length)
        log("train_data", examples=len(texts), dropped_too_long=dropped, rung=attempt.rung)
        trainer = _build_trainer(config, attempt, model, tokenizer, texts, run_dir, smoke)
        existing = [] if smoke else list_checkpoints(run_dir)
        resume = existing[-1] if existing else None
        if resume is not None:
            log("resume", checkpoint=resume.name)
        output_obj = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if not smoke:
            trainer.save_model(str(run_dir / FINAL_NAME))
        result = AttemptResult(
            attempt=attempt,
            ok=True,
            runtime_s=round(time.perf_counter() - started, 1),
            peak_vram_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            train_loss=float(getattr(output_obj, "training_loss", float("nan"))),
            global_step=int(getattr(output_obj, "global_step", 0)),
            dropped_too_long=dropped,
        )
        trainer = None
        return result, model, tokenizer
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        log("oom", rung=attempt.rung, detail=type(exc).__name__)
    # Out of memory: drop every reference outside the except block, then free.
    trainer = model = tokenizer = None
    _free_gpu()
    oom = AttemptResult(
        attempt=attempt,
        ok=False,
        oom=True,
        runtime_s=round(time.perf_counter() - started, 1),
        peak_vram_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    )
    return oom, None, None


def _generate(
    model: Any, tokenizer: Any, items: Sequence[Item], system: str, max_new: int
) -> list[str]:
    import torch

    outputs: list[str] = []
    for item in items:
        messages = build_messages(item, system=system, include_target=False)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        prompt_len = int(inputs["input_ids"].shape[1])
        outputs.append(str(tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)))
    return outputs


def _val_metrics(items: Sequence[Item], outputs: Sequence[str]) -> dict[str, float | int | None]:
    return score_examples(
        [
            ScoredExample(
                gold_answerable=item.target.answerable,
                gold_concepts=tuple(item.target.concepts),
                passage=item.passage,
                output=text,
            )
            for item, text in zip(items, outputs, strict=True)
        ]
    )


def _select_and_save(
    config: TrainConfig,
    attempt: Attempt,
    model: Any,
    tokenizer: Any,
    data: PreparedData,
    system: str,
) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from unsloth import FastLanguageModel

    run_dir = config.output.checkpoints() / f"rung-{attempt.rung}"
    candidates = selection_candidates(run_dir, config.selection.max_candidates)
    if not candidates:
        raise RuntimeError(f"no checkpoints to select from in {run_dir}")
    FastLanguageModel.for_inference(model)
    scores: dict[str, float | None] = {}
    metrics: dict[str, dict[str, float | int | None]] = {}
    for candidate in candidates:
        state = load_file(str(candidate / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, state)
        outputs = _generate(
            model, tokenizer, data.val_subset, system, config.selection.max_new_tokens
        )
        metrics[candidate.name] = _val_metrics(data.val_subset, outputs)
        f1 = metrics[candidate.name]["answerability_f1"]
        scores[candidate.name] = None if f1 is None else float(f1)
        log("val_score", checkpoint=candidate.name, answerability_f1=scores[candidate.name])
    best = pick_best(scores)
    best_state = load_file(str(run_dir / best / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, best_state)
    adapter_dir = config.output.adapter()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {"best_checkpoint": best, "val_scores": scores, "val_metrics": metrics[best]}


def _push_to_hub(config: TrainConfig, summary: dict[str, Any], train_sources: set[str]) -> bool:
    token = hf_token()
    if not config.hub.push or token is None:
        log("hub_push_skipped", reason="disabled" if not config.hub.push else "no HF_TOKEN")
        return False
    from huggingface_hub import HfApi

    from app.train.model_card import build_model_card

    adapter_dir = config.output.adapter()
    card = build_model_card(
        base_model=str(summary["base_model"]),
        repo_id=config.hub.repo_id,
        train_source_ids=train_sources,
        summary={k: v for k, v in summary.items() if k != "val_scores"},
    )
    (adapter_dir / "README.md").write_text(card, encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(config.hub.repo_id, private=config.hub.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=config.hub.repo_id,
        commit_message=f"Adapter from {summary.get('best_checkpoint')}",
    )
    log("hub_push", repo_id=config.hub.repo_id)
    return True


def run(config: TrainConfig, data_path: Path, *, smoke: bool, forced_rung: int | None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    base_model = get_base_model()
    data = prepare_data(data_path, config)
    log(
        "data",
        path=str(data_path),
        counts=data.splits.counts(),
        val_subset=len(data.val_subset),
        test_split_sha256=data.test_split_sha256,
    )
    system = build_system_prompt(version=config.prompt.answer_system_version)
    ladder = attempt_ladder(config)
    state_path = config.output.state()
    state = read_state(state_path)
    rung = start_rung(state, forced_rung, len(ladder))

    result: AttemptResult | None = None
    model = tokenizer = None
    for attempt in ladder[rung:]:
        log("attempt", smoke=smoke, **attempt.model_dump())
        previous = state.get("full") if state.get("rung") == attempt.rung else None
        result, model, tokenizer = _run_attempt(
            config, attempt, data, system, base_model, smoke=smoke, previous=previous
        )
        state.update({"rung": attempt.rung, "updated_at": datetime.now(UTC).isoformat()})
        if result.ok:
            record = {
                "runtime_s": result.runtime_s,
                "peak_vram_gib": result.peak_vram_gib,
                "peak_alloc_gib": result.peak_alloc_gib,
                "train_loss": result.train_loss,
                "global_step": result.global_step,
                "dropped_too_long": result.dropped_too_long,
            }
            state["smoke" if smoke else "full"] = record
            write_state(state_path, state)
            break
        if attempt.rung + 1 < len(ladder):
            state["rung"] = attempt.rung + 1
        write_state(state_path, state)
    if result is None or not result.ok:
        log("failed", reason="out of memory on every rung of the OOM ladder")
        return 3

    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "smoke": smoke,
        "base_model": base_model,
        "gpu": _gpu_name(),
        "rung": result.attempt.rung,
        "per_device_train_batch_size": result.attempt.per_device_train_batch_size,
        "gradient_accumulation_steps": result.attempt.gradient_accumulation_steps,
        "max_seq_length": result.attempt.max_seq_length,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.num_train_epochs,
        "n_train": len(data.train) - result.dropped_too_long,
        "n_val": len(data.splits.val),
        "n_test": len(data.splits.test),
        "dropped_too_long": result.dropped_too_long,
        "global_step": result.global_step,
        "train_loss": result.train_loss,
        "train_runtime_s": result.runtime_s,
        "peak_vram_reserved_gib": result.peak_vram_gib,
        "peak_vram_allocated_gib": result.peak_alloc_gib,
        "dataset_sha256": data.dataset_sha256,
        "test_split_sha256": data.test_split_sha256,
    }
    if not smoke:
        summary.update(_select_and_save(config, result.attempt, model, tokenizer, data, system))
        summary["hub_pushed"] = _push_to_hub(
            config, summary, {item.source_id for item in data.train}
        )
    out = config.output.working() / ("smoke_summary.json" if smoke else config.output.summary_file)
    write_state(out, summary)
    log("summary", path=str(out), **{k: v for k, v in summary.items() if k != "val_scores"})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.train.qlora")
    parser.add_argument("--config", type=Path, default=None, help="path to train.yaml")
    parser.add_argument("--data", type=str, default=None, help="path to reviewed.jsonl")
    parser.add_argument("--smoke", action="store_true", help="short run; logs peak VRAM")
    parser.add_argument("--working-dir", type=str, default=None, help="override output dir")
    parser.add_argument("--rung", type=int, default=None, help="force an OOM-ladder rung")
    parser.add_argument("--no-push", action="store_true", help="never push to the HF Hub")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="validate the dataset and split (CPU only), print counts, and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    if args.working_dir:
        config.output.working_dir = args.working_dir
    if args.no_push:
        config.hub.push = False
    data_path = find_reviewed_jsonl(
        args.data or config.data.path or None,
        search_roots=config.data.search_roots,
        filename=config.data.filename,
    )
    if args.check_data:
        data = prepare_data(data_path, config)
        print(
            json.dumps(
                {
                    "path": str(data_path),
                    "counts": data.splits.counts(),
                    "val_subset": len(data.val_subset),
                    "dataset_sha256": data.dataset_sha256,
                    "test_split_sha256": data.test_split_sha256,
                    "no_test_in_train": True,
                },
                indent=2,
            )
        )
        return 0
    return run(config, data_path, smoke=args.smoke, forced_rung=args.rung)


if __name__ == "__main__":
    sys.exit(main())
