"""Generate grounded-QA items via GuardedBedrockClient."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.bedrock import BedrockResponse, FakeTransport, GuardedBedrockClient
from app.budget import BudgetGuard, estimate_cost
from app.data.chunk import Chunk, count_tokens, load_data_config
from app.model_config import get_model
from app.paths import repo_root
from app.prompts import load_prompt
from app.schema import Item, load_concept_vocabulary

JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class DryRunEstimate:
    call_count: int
    worst_case_cost_usd: Decimal
    model_id: str
    max_tokens: int
    estimated_input_tokens_per_call: int


def _parse_json_payload(text: str) -> Any:
    stripped = text.strip()
    fence = JSON_FENCE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    return json.loads(stripped)


def build_generation_prompt(
    chunk: Chunk,
    *,
    vocabulary: Sequence[str],
    items_per_passage: int,
    unanswerable_share: float,
) -> tuple[str, str]:
    system = load_prompt("generate_items", 1)
    user = (
        f"section_id: {chunk.section_id}\n"
        f"source_id: {chunk.source_id}\n"
        f"items_requested: {items_per_passage}\n"
        f"unanswerable_share: {unanswerable_share}\n"
        f"concept_vocabulary:\n- " + "\n- ".join(vocabulary) + "\n\npassage:\n" + chunk.text
    )
    return system, user


def project_dry_run(
    chunks: Sequence[Chunk],
    *,
    guard: BudgetGuard | None = None,
    over_generation: float | None = None,
) -> DryRunEstimate:
    config = load_data_config()
    gen_cfg = config.get("generation", {})
    over = (
        over_generation
        if over_generation is not None
        else float(gen_cfg.get("over_generation", 1.4))
    )
    items_per = int(gen_cfg.get("items_per_passage", 5))
    model = get_model("generator")
    # One call per selected chunk; over-generation selects more chunks.
    n_chunks = max(1, int(round(len(chunks) * over))) if chunks else 0
    selected = list(chunks)[:n_chunks] if n_chunks else []
    if not selected and chunks:
        selected = list(chunks)
    vocab = sorted(load_concept_vocabulary())
    unans = float(gen_cfg.get("unanswerable_share", 0.2))
    input_estimates: list[int] = []
    for chunk in selected:
        system, user = build_generation_prompt(
            chunk,
            vocabulary=vocab,
            items_per_passage=items_per,
            unanswerable_share=unans,
        )
        input_estimates.append(count_tokens(system) + count_tokens(user))
    avg_input = int(sum(input_estimates) / len(input_estimates)) if input_estimates else 0
    worst = Decimal("0")
    for est in input_estimates:
        worst += estimate_cost(model.id, est, model.max_tokens)
        if guard is not None:
            # Validate the projected call would reserve under caps (no commit).
            reservation = guard.reserve("generation", model.id, est, model.max_tokens)
            guard.release(reservation)
    return DryRunEstimate(
        call_count=len(selected),
        worst_case_cost_usd=worst,
        model_id=model.id,
        max_tokens=model.max_tokens,
        estimated_input_tokens_per_call=avg_input,
    )


def _items_from_response(
    response: BedrockResponse,
    *,
    chunk: Chunk,
    split: str,
    licence_class: str,
) -> list[Item]:
    payload = _parse_json_payload(response.text)
    if isinstance(payload, dict) and "items" in payload:
        raw_items = payload["items"]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = [payload]
    items: list[Item] = []
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        data = dict(raw)
        data.setdefault("source_id", chunk.source_id)
        data.setdefault("section_id", chunk.section_id)
        data.setdefault("passage", chunk.text)
        data.setdefault("split", split)
        data.setdefault("licence_class", licence_class)
        data.setdefault(
            "id",
            f"{chunk.section_id.replace('/', '-')}-p{chunk.chunk_index:03d}-q{index}",
        )
        try:
            items.append(Item.model_validate(data))
        except Exception:
            continue
    return items


def generate_items(
    chunks: Sequence[tuple[Chunk, str, str]],
    client: GuardedBedrockClient,
    *,
    dry_run: bool = False,
) -> list[Item] | DryRunEstimate:
    """chunks entries are (chunk, split, licence_class)."""
    plain_chunks = [entry[0] for entry in chunks]
    if dry_run:
        return project_dry_run(plain_chunks, guard=client.guard)

    config = load_data_config()
    gen_cfg = config.get("generation", {})
    items_per = int(gen_cfg.get("items_per_passage", 5))
    unans = float(gen_cfg.get("unanswerable_share", 0.2))
    model = get_model("generator")
    vocab = sorted(load_concept_vocabulary())
    out: list[Item] = []
    for chunk, split, licence_class in chunks:
        system, user = build_generation_prompt(
            chunk,
            vocabulary=vocab,
            items_per_passage=items_per,
            unanswerable_share=unans,
        )
        messages = [{"role": "user", "content": [{"text": user}]}]
        response = client.converse(
            phase="generation",
            model_id=model.id,
            messages=messages,
            system=system,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            input_tokens_est=count_tokens(system) + count_tokens(user),
        )
        out.extend(
            _items_from_response(
                response,
                chunk=chunk,
                split=split,
                licence_class=licence_class,
            )
        )
    return out


def make_fake_client(guard: BudgetGuard, text: str = '{"items": []}') -> GuardedBedrockClient:
    return GuardedBedrockClient(guard, FakeTransport(text=text))


def default_chunks_path() -> Path:
    return repo_root() / "data" / "private" / "chunks.jsonl"
