"""Shared task format: passage + question in, strict JSON target out.

Training, local GPU eval and the Bedrock baseline all build prompts here so the
three systems see exactly the same instructions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.prompts import load_prompt
from app.schema import Item, Target, load_concept_vocabulary

ANSWER_PROMPT_NAME = "answer_system"
ANSWER_PROMPT_VERSION = 1


def build_system_prompt(
    *,
    version: int = ANSWER_PROMPT_VERSION,
    vocabulary: Sequence[str] | None = None,
) -> str:
    """Answer system prompt plus the fixed concept vocabulary."""
    base = load_prompt(ANSWER_PROMPT_NAME, version).strip()
    tags = sorted(vocabulary) if vocabulary is not None else sorted(load_concept_vocabulary())
    return base + "\n\nConcept vocabulary (use only these tags):\n" + ", ".join(tags)


def build_user_prompt(passage: str, question: str) -> str:
    return f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nReply with the JSON object only."


def target_json(target: Target) -> str:
    """Canonical assistant completion for a target."""
    payload: dict[str, Any] = {
        "answerable": target.answerable,
        "answer": target.answer,
        "quote": target.quote,
        "concepts": list(target.concepts),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    item: Item,
    *,
    system: str,
    include_target: bool,
) -> list[dict[str, str]]:
    """OpenAI/HF-style chat messages for one item."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(item.passage, item.question)},
    ]
    if include_target:
        messages.append({"role": "assistant", "content": target_json(item.target)})
    return messages
