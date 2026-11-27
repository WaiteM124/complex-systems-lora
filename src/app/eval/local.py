"""Test-split predictions for (a) base and (b) base + LoRA on one GPU.

Run on Kaggle (T4): `python -m app.eval.local --system base` then
`python -m app.eval.local --system lora [--adapter PATH_OR_HUB_REPO]`.

transformers + PEFT, greedy decoding, batch size 1 so per-item latency is honest.
Predictions are appended per item (resumable) to
`/kaggle/working/predictions/<system>.jsonl`; they hold model text and stay
private. Score them with `python -m app.eval.score --kaggle --run <name>`.

torch/transformers/peft are imported lazily inside functions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from app.data.reviewed import load_items, split_items
from app.eval.config import AdapterConfig, EvalConfig, load_eval_config, resolve_data_path
from app.eval.predictions import Prediction, append_jsonl, done_ids, predictions_path, read_jsonl
from app.model_config import get_base_model
from app.paths import repo_root
from app.schema import Item
from app.task import build_messages, build_system_prompt

System = Literal["base", "lora"]
ADAPTER_MARKER = "adapter_config.json"


def resolve_adapter(cli_value: str | None, config: AdapterConfig) -> str:
    """--adapter, then the local adapter dir, then a Kaggle input glob, then the Hub repo."""
    if cli_value:
        return cli_value
    local = Path(config.local_dir)
    if (local / ADAPTER_MARKER).is_file():
        return str(local)
    for root in config.search_roots:
        root_path = Path(root)
        if root_path.is_dir():
            found = sorted(root_path.glob(f"**/{ADAPTER_MARKER}"))
            if len(found) == 1:
                return str(found[0].parent)
            if len(found) > 1:
                listed = ", ".join(str(p.parent) for p in found)
                raise FileNotFoundError(f"several adapters found ({listed}); pass --adapter")
    return config.hub_repo


def pending_items(test: Sequence[Item], done: set[str], limit: int | None) -> list[Item]:
    todo = [item for item in sorted(test, key=lambda i: i.id) if item.id not in done]
    return todo[:limit] if limit is not None else todo


def _load(system: System, adapter: str | None, config: EvalConfig) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = get_base_model()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    kwargs: dict[str, Any] = {"device_map": {"": 0}}
    if config.local.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["dtype"] = getattr(torch, config.local.dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if system == "lora":
        from peft import PeftModel

        if adapter is None:
            raise ValueError("lora system needs an adapter")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    hardware = (
        str(torch.cuda.get_device_name(0))
        if torch.cuda.is_available()
        else config.local.hardware_label_default
    )
    return model, tokenizer, hardware


def generate_one(
    model: Any, tokenizer: Any, item: Item, system_prompt: str, max_new_tokens: int
) -> tuple[str, float, int, int]:
    import torch

    messages = build_messages(item, system=system_prompt, include_target=False)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    new_tokens = out[0, prompt_len:]
    text = str(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return text, latency_ms, prompt_len, int(new_tokens.shape[0])


def run(
    *,
    system: System,
    data_path: Path,
    out_dir: Path,
    adapter: str | None,
    config: EvalConfig,
    limit: int | None,
) -> Path:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    test = split_items(load_items(data_path)).test
    out_path = predictions_path(out_dir, system)
    todo = pending_items(test, done_ids(read_jsonl(out_path, Prediction)), limit)
    print(json.dumps({"system": system, "test_items": len(test), "todo": len(todo)}), flush=True)
    if not todo:
        return out_path
    model, tokenizer, hardware = _load(system, adapter, config)
    prompt = build_system_prompt(version=config.prompt.answer_system_version)
    base_model = get_base_model()
    for index, item in enumerate(todo, start=1):
        text, latency_ms, n_in, n_out = generate_one(
            model, tokenizer, item, prompt, config.local.max_new_tokens
        )
        record = Prediction(
            id=item.id,
            system=system,
            output=text,
            model=base_model,
            adapter=adapter if system == "lora" else None,
            hardware=hardware,
            latency_ms=round(latency_ms, 1),
            input_tokens=n_in,
            output_tokens=n_out,
        )
        append_jsonl(out_path, [record])
        if index % 10 == 0 or index == len(todo):
            print(f"{system}: {index}/{len(todo)}", flush=True)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.eval.local")
    parser.add_argument("--system", choices=["base", "lora"], required=True)
    parser.add_argument("--data", default=None, help="path to reviewed.jsonl")
    parser.add_argument("--adapter", default=None, help="adapter dir or HF Hub repo id")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only N items (debug)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_eval_config()
    data_path = resolve_data_path(args.data, config, private_dir=repo_root() / "data" / "private")
    adapter = resolve_adapter(args.adapter, config.adapter) if args.system == "lora" else None
    out_dir = args.out_dir or Path(config.paths.kaggle_predictions_dir)
    path = run(
        system=args.system,
        data_path=data_path,
        out_dir=out_dir,
        adapter=adapter,
        config=config,
        limit=args.limit,
    )
    print(json.dumps({"predictions": str(path), "adapter": adapter}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
