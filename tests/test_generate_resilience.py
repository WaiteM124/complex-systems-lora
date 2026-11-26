"""Generation robustness: bad replies, retry/skip, resume, and budget across runs.

All tests use scripted fake transports. No AWS or Bedrock calls.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.bedrock import BedrockResponse, GuardedBedrockClient
from app.budget import BudgetGuard, clear_budget_caches
from app.data.chunk import Chunk, chunk_hash
from app.data.filters import filter_items
from app.data.generate import (
    GenerationPaths,
    ReplyParseError,
    _parse_json_payload,
    chunk_key,
    run_generation,
)
from app.schema import Item

PASSAGE = (
    "Emergence describes system-level patterns that arise from local interactions. "
    'Some authors call this "more is different". Feedback loops can amplify small changes.'
)
QUOTE = "Emergence describes system-level patterns that arise from local interactions."


def _good_reply(question: str = "What does emergence describe?") -> str:
    return json.dumps(
        {
            "items": [
                {
                    "id": "q1",  # model-supplied id must be overridden
                    "split": "train",  # model must not choose the split
                    "question": question,
                    "target": {
                        "answerable": True,
                        "answer": "System-level patterns from local interactions.",
                        "quote": QUOTE,
                        "concepts": ["emergence"],
                    },
                },
                {
                    "question": "Who coined the term in 1850?",
                    "target": {"answerable": False, "answer": "", "quote": "", "concepts": []},
                },
            ]
        }
    )


# Unescaped inner quotes: the observed failure mode ("Expecting ',' delimiter").
BAD_QUOTES = (
    '{"items":[{"question":"Why do some say "more is different"?","target":'
    '{"answerable":false,"answer":"","quote":"","concepts":[]}}]}'
)
GARBAGE = "I'm sorry, here are your items: items items items"


class ScriptedTransport:
    """Returns the scripted replies in order, then repeats the last one."""

    def __init__(self, replies: list[str], *, output_tokens: int = 200) -> None:
        self.replies = replies
        self.output_tokens = output_tokens
        self.calls: list[dict[str, Any]] = []

    def converse(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> BedrockResponse:
        index = min(len(self.calls), len(self.replies) - 1)
        self.calls.append({"model_id": model_id, "max_tokens": max_tokens})
        return BedrockResponse(
            text=self.replies[index], input_tokens=500, output_tokens=self.output_tokens
        )


class PerChunkTransport(ScriptedTransport):
    """Fails for passages containing a marker, succeeds otherwise."""

    def __init__(self, bad_marker: str) -> None:
        super().__init__([_good_reply()])
        self.bad_marker = bad_marker

    def converse(self, **kwargs: Any) -> BedrockResponse:
        text = kwargs["messages"][0]["content"][0]["text"]
        self.calls.append({"model_id": kwargs["model_id"]})
        reply = GARBAGE if self.bad_marker in text else _good_reply()
        return BedrockResponse(text=reply, input_tokens=500, output_tokens=200)


@pytest.fixture(autouse=True)
def _clear() -> None:
    clear_budget_caches()


def _chunk(index: int, *, text: str = PASSAGE) -> Chunk:
    section_path = f"Fixture > Section {index}"
    return Chunk(
        section_path=section_path,
        section_id=f"fixture/s{index:02d}",
        text=text,
        token_count=len(text.split()),
        chunk_hash=chunk_hash(text, section_path),
        chunk_index=index,
        source_id="cs-003",
    )


def _entries(n: int) -> list[tuple[Chunk, str, str]]:
    return [(_chunk(i), "test" if i == 0 else "train", "A") for i in range(n)]


def _client(tmp_path: Path, transport: Any, **guard_kwargs: Any) -> GuardedBedrockClient:
    guard = BudgetGuard(ledger_path=tmp_path / "ledger.jsonl", **guard_kwargs)
    return GuardedBedrockClient(guard, transport)


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------------ parsing


def test_parser_repairs_unescaped_inner_quotes() -> None:
    with pytest.raises(json.JSONDecodeError, match="Expecting ',' delimiter"):
        json.loads(BAD_QUOTES)
    payload = _parse_json_payload(BAD_QUOTES)
    assert payload["items"][0]["question"] == 'Why do some say "more is different"?'


def test_parser_repairs_latex_backslashes_and_fences() -> None:
    reply = '```json\n{"items":[{"question":"Is $\\alpha \\in \\{0,1\\}$?"}]}\n```'
    payload = _parse_json_payload(reply)
    assert payload["items"][0]["question"] == "Is $\\alpha \\in \\{0,1\\}$?"


def test_parser_salvages_items_from_truncated_reply() -> None:
    full = _good_reply()
    truncated = full[: full.index("Who coined") + 5]
    payload = _parse_json_payload(truncated)
    assert len(payload["items"]) == 1


def test_parser_raises_reply_parse_error_on_garbage() -> None:
    with pytest.raises(ReplyParseError):
        _parse_json_payload(GARBAGE)


# ------------------------------------------------------------------ retry / skip


def test_malformed_then_success_on_retry(tmp_path: Path) -> None:
    transport = ScriptedTransport([GARBAGE, _good_reply()])
    client = _client(tmp_path, transport)
    paths = GenerationPaths.in_dir(tmp_path / "private")
    summary = run_generation(_entries(1), client, paths)
    assert len(transport.calls) == 2
    assert summary.chunks_ok == 1
    assert summary.retried == 1
    assert summary.skipped == 0
    items = _lines(paths.generated)
    assert len(items) == 2
    # Provenance and ids come from the pipeline, not the model.
    assert items[0]["id"] == "fixture-s00-p000-q1"
    assert items[0]["split"] == "test"
    assert not paths.failures.exists()
    # Two calls were paid for and both are in the persisted ledger.
    assert summary.spend_this_run_usd == client.guard.spent("generation")
    assert len(_lines(tmp_path / "ledger.jsonl")) == 2


def test_malformed_twice_is_skipped_and_recorded(tmp_path: Path) -> None:
    transport = PerChunkTransport(bad_marker="BADCHUNK")
    entries = _entries(3)
    bad = _chunk(1, text=PASSAGE + " BADCHUNK")
    entries[1] = (bad, "train", "A")
    client = _client(tmp_path, transport)
    paths = GenerationPaths.in_dir(tmp_path / "private")
    summary = run_generation(entries, client, paths)
    assert len(transport.calls) == 4  # 1 + 2 (retry) + 1
    assert summary.chunks_ok == 2
    assert summary.skipped == 1
    assert summary.retried == 1
    assert summary.items_generated == 4
    failures = _lines(paths.failures)
    assert len(failures) == 1
    assert failures[0]["chunk_key"] == chunk_key(bad)
    assert failures[0]["attempts"] == 2
    assert "invalid JSON" in failures[0]["reason"]
    # A rerun does not pay for the skipped chunk again unless asked.
    again = run_generation(entries, client, paths)
    assert again.already_done == 3
    assert len(transport.calls) == 4
    retry = run_generation(entries, client, paths, retry_failed=True)
    assert retry.already_done == 2
    assert len(transport.calls) == 6


# ------------------------------------------------------------------ resume


class InterruptAfter(ScriptedTransport):
    def __init__(self, n: int) -> None:
        super().__init__([_good_reply()])
        self.n = n

    def converse(self, **kwargs: Any) -> BedrockResponse:
        if len(self.calls) >= self.n:
            raise KeyboardInterrupt
        return super().converse(**kwargs)


def test_resume_skips_completed_chunks(tmp_path: Path) -> None:
    entries = _entries(5)
    paths = GenerationPaths.in_dir(tmp_path / "private")
    first = run_generation(entries, _client(tmp_path, InterruptAfter(2)), paths)
    assert first.stopped.startswith("interrupted")
    assert first.chunks_ok == 2
    assert len(_lines(paths.generated)) == 4
    # Simulate a torn write: a partial line from a crash mid-append.
    with paths.generated.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "fixture-s02-p002-q1", "trunc')
    transport = ScriptedTransport([_good_reply()])
    second = run_generation(entries, _client(tmp_path, transport), paths)
    assert second.already_done == 2
    assert second.chunks_ok == 3
    assert len(transport.calls) == 3
    items = _lines(paths.generated)
    assert len(items) == 10
    assert len({item["id"] for item in items}) == 10
    # Downstream filter reads the appended file.
    filtered = filter_items(items)
    assert filtered and all(isinstance(item, Item) for item in filtered)


def test_budget_holds_across_resumed_run(tmp_path: Path) -> None:
    """Cap is enforced from the persisted ledger, so a new run cannot reset it."""
    entries = _entries(50)
    paths = GenerationPaths.in_dir(tmp_path / "private")
    cap = 0.03
    first = run_generation(
        entries[:10],
        _client(tmp_path, ScriptedTransport([_good_reply()]), total_cap_override=cap),
        paths,
    )
    spent_after_first = first.generation_spent_total_usd
    assert spent_after_first > 0
    # New guard and client, as on a rerun after a crash; malformed replies force retries.
    transport = ScriptedTransport([GARBAGE, _good_reply()], output_tokens=1200)
    second = run_generation(
        entries,
        _client(tmp_path, transport, total_cap_override=cap),
        paths,
    )
    assert second.already_done == 10
    assert second.stopped.startswith("budget cap reached")
    ledger_total = sum(Decimal(str(e["cost_usd"])) for e in _lines(tmp_path / "ledger.jsonl"))
    assert ledger_total <= Decimal(str(cap))
    assert second.generation_spent_total_usd == ledger_total
    assert second.generation_spent_total_usd > spent_after_first
