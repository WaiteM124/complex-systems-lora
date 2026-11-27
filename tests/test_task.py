from __future__ import annotations

import json

from _synthetic import make_item
from app.schema import Target, load_concept_vocabulary
from app.task import build_messages, build_system_prompt, build_user_prompt, target_json


def test_system_prompt_has_rules_and_full_vocabulary() -> None:
    prompt = build_system_prompt()
    assert "strict JSON" in prompt
    for tag in load_concept_vocabulary():
        assert tag in prompt


def test_user_prompt_holds_passage_and_question() -> None:
    text = build_user_prompt("P text.", "Q?")
    assert "P text." in text and "Q?" in text


def test_target_json_round_trips() -> None:
    item = make_item(0, "train")
    parsed = json.loads(target_json(item.target))
    assert list(parsed) == ["answerable", "answer", "quote", "concepts"]
    assert Target.model_validate(parsed) == item.target


def test_messages_with_and_without_target() -> None:
    item = make_item(1, "train")
    with_target = build_messages(item, system="S", include_target=True)
    assert [m["role"] for m in with_target] == ["system", "user", "assistant"]
    assert with_target[-1]["content"] == target_json(item.target)
    prompt_only = build_messages(item, system="S", include_target=False)
    assert [m["role"] for m in prompt_only] == ["system", "user"]
