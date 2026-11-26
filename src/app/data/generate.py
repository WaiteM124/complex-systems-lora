"""Generate grounded-QA items via GuardedBedrockClient.

Live generation is resumable: each finished chunk is appended to
``generated.jsonl`` and recorded in ``generate_progress.jsonl`` before the next
call, so a crash or Ctrl-C keeps finished work and a rerun skips it. A reply
that cannot be parsed is retried once, then skipped and logged to
``generate_failures.jsonl``. Spend is enforced per call by the persisted
BudgetGuard ledger, so caps hold across retries and resumed runs.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.bedrock import BedrockResponse, FakeTransport, GuardedBedrockClient
from app.budget import BudgetExceeded, BudgetGuard, estimate_cost
from app.data.chunk import Chunk, count_tokens, load_data_config
from app.model_config import get_model
from app.paths import repo_root
from app.prompts import load_prompt
from app.schema import Item, load_concept_vocabulary

PROMPT_NAME = "generate_items"
PROMPT_VERSION = 2
MAX_ATTEMPTS = 2  # first call plus at most one retry per chunk

JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# A backslash that does not start a valid JSON escape (e.g. LaTeX `\alpha`, `\{`).
INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')
_RESTORE_ESCAPES = {"\b": "\\b", "\f": "\\f", "\t": "\\t", "\n": "\\n", "\r": "\\r"}


class ReplyParseError(ValueError):
    """Raised when a model reply contains no usable JSON items."""


@dataclass(frozen=True)
class DryRunEstimate:
    call_count: int
    worst_case_cost_usd: Decimal
    model_id: str
    max_tokens: int
    estimated_input_tokens_per_call: int


# --------------------------------------------------------------------------- parsing


def _strip_wrappers(text: str) -> str:
    stripped = THINK_BLOCK.sub("", text).strip()
    fence = JSON_FENCE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        stripped = stripped[start : end + 1] if end > start else stripped[start:]
    return stripped


def _escape_invalid_backslashes(text: str) -> str:
    return INVALID_ESCAPE.sub(r"\\\\", text)


def _escape_inner_quotes(text: str) -> str:
    """Escape `"` characters inside JSON strings that do not close the string.

    A quote closes a string only when the next non-space character is `:`, `}`,
    `]`, end of input, or a `,` that is itself followed by `"`, `{`, `[`, `}` or `]`.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)

    def next_non_space(pos: int) -> int:
        while pos < n and text[pos] in " \t\r\n":
            pos += 1
        return pos

    while i < n:
        char = text[i]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue
        if char == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        if char == '"':
            j = next_non_space(i + 1)
            closes = j >= n or text[j] in ":}]"
            if not closes and text[j] == ",":
                k = next_non_space(j + 1)
                closes = k >= n or text[k] in '"{[}]'
            if closes:
                out.append('"')
                in_string = False
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _candidates(text: str) -> list[str]:
    base = _strip_wrappers(text)
    backslash = _escape_invalid_backslashes(base)
    quotes = _escape_inner_quotes(base)
    both = _escape_inner_quotes(backslash)
    seen: list[str] = []
    for candidate in (base, backslash, quotes, both):
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _salvage_items(text: str) -> list[Any]:
    """Decode complete elements of the items array, stopping at the first bad one.

    Recovers the finished items from a reply truncated at max_tokens.
    """
    decoder = json.JSONDecoder(strict=False)
    match = re.search(r'"items"\s*:\s*\[', text)
    if match:
        pos = match.end()
    elif text.lstrip().startswith("["):
        pos = text.index("[") + 1
    else:
        return []
    out: list[Any] = []
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or text[pos] == "]":
            break
        try:
            value, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        out.append(value)
    return out


def _parse_json_payload(text: str) -> Any:
    """Parse a model reply, repairing common defects. Raises ReplyParseError."""
    if not text or not text.strip():
        raise ReplyParseError("empty reply")
    first_error: json.JSONDecodeError | None = None
    candidates = _candidates(text)
    for candidate in candidates:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
    for candidate in candidates:
        salvaged = _salvage_items(candidate)
        if any(isinstance(value, dict) for value in salvaged):
            return {"items": salvaged, "_salvaged": True}
    detail = f"{first_error.msg} at char {first_error.pos}" if first_error else "no JSON found"
    raise ReplyParseError(f"invalid JSON: {detail}")


def _restore_escapes(value: str) -> str:
    return "".join(_RESTORE_ESCAPES.get(char, char) for char in value)


def _fix_latex_controls(data: dict[str, Any], passage: str) -> None:
    """Undo LaTeX commands that JSON decoded as control characters (`\\beta` -> backspace)."""
    for key in ("question",):
        if isinstance(data.get(key), str):
            data[key] = data[key].replace("\b", "\\b").replace("\f", "\\f")
    target = data.get("target")
    if not isinstance(target, dict):
        return
    for key in ("answer",):
        if isinstance(target.get(key), str):
            target[key] = target[key].replace("\b", "\\b").replace("\f", "\\f")
    quote = target.get("quote")
    if isinstance(quote, str) and quote and quote not in passage:
        restored = _restore_escapes(quote)
        if restored in passage:
            target["quote"] = restored


# --------------------------------------------------------------------------- prompts


def build_generation_prompt(
    chunk: Chunk,
    *,
    vocabulary: Sequence[str],
    items_per_passage: int,
    unanswerable_share: float,
) -> tuple[str, str]:
    system = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    user = (
        f"section_id: {chunk.section_id}\n"
        f"source_id: {chunk.source_id}\n"
        f"items_requested: {items_per_passage}\n"
        f"unanswerable_share: {unanswerable_share}\n"
        f"concept_vocabulary:\n- " + "\n- ".join(vocabulary) + "\n\npassage:\n" + chunk.text
    )
    return system, user


def reservation_input_tokens(system: str, user: str) -> int:
    """Conservative input-token estimate used for budget reservations.

    Whitespace words undercount model tokens on maths-heavy text, so take the
    larger of words and characters/3. The ledger records measured usage.
    """
    text = system + "\n" + user
    return max(count_tokens(system) + count_tokens(user), math.ceil(len(text) / 3))


def _generation_settings() -> tuple[int, float, list[str]]:
    gen_cfg = load_data_config().get("generation", {})
    items_per = int(gen_cfg.get("items_per_passage", 5))
    unans = float(gen_cfg.get("unanswerable_share", 0.2))
    return items_per, unans, sorted(load_concept_vocabulary())


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
    items_per, unans, vocab = _generation_settings()
    model = get_model("generator")
    # One call per selected chunk; over-generation selects more chunks.
    n_chunks = max(1, int(round(len(chunks) * over))) if chunks else 0
    selected = list(chunks)[:n_chunks] if n_chunks else []
    if not selected and chunks:
        selected = list(chunks)
    input_estimates: list[int] = []
    for chunk in selected:
        system, user = build_generation_prompt(
            chunk,
            vocabulary=vocab,
            items_per_passage=items_per,
            unanswerable_share=unans,
        )
        input_estimates.append(reservation_input_tokens(system, user))
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


# --------------------------------------------------------------------------- items


def chunk_key(chunk: Chunk) -> str:
    return f"{chunk.section_id}#{chunk.chunk_index}#{chunk.chunk_hash[:16]}"


def _item_id(chunk: Chunk, index: int) -> str:
    return f"{chunk.section_id.replace('/', '-')}-p{chunk.chunk_index:03d}-q{index}"


def _items_from_payload(
    payload: Any,
    *,
    chunk: Chunk,
    split: str,
    licence_class: str,
) -> list[Item]:
    if isinstance(payload, dict) and "items" in payload:
        raw_items = payload["items"]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = [payload]
    if not isinstance(raw_items, list):
        raw_items = [raw_items]
    items: list[Item] = []
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        data = dict(raw)
        # Provenance comes from the pipeline, never from the model.
        data["id"] = _item_id(chunk, index)
        data["source_id"] = chunk.source_id
        data["section_id"] = chunk.section_id
        data["passage"] = chunk.text
        data["split"] = split
        data["licence_class"] = licence_class
        _fix_latex_controls(data, chunk.text)
        try:
            items.append(Item.model_validate(data))
        except Exception:
            continue
    return items


def _items_from_response(
    response: BedrockResponse,
    *,
    chunk: Chunk,
    split: str,
    licence_class: str,
) -> list[Item]:
    payload = _parse_json_payload(response.text)
    return _items_from_payload(payload, chunk=chunk, split=split, licence_class=licence_class)


@dataclass
class ChunkOutcome:
    status: str  # "ok" or "skipped"
    items: list[Item]
    attempts: int
    cost_usd: Decimal
    reason: str = ""


def generate_for_chunk(
    chunk: Chunk,
    split: str,
    licence_class: str,
    client: GuardedBedrockClient,
) -> ChunkOutcome:
    """One call, plus at most one retry when the reply cannot be parsed.

    BudgetExceeded and transport errors propagate; finished chunks are already saved.
    """
    items_per, unans, vocab = _generation_settings()
    model = get_model("generator")
    system, user = build_generation_prompt(
        chunk,
        vocabulary=vocab,
        items_per_passage=items_per,
        unanswerable_share=unans,
    )
    messages = [{"role": "user", "content": [{"text": user}]}]
    cost = Decimal("0")
    reasons: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.converse(
            phase="generation",
            model_id=model.id,
            messages=messages,
            system=system,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            input_tokens_est=reservation_input_tokens(system, user),
        )
        cost += estimate_cost(model.id, response.input_tokens, response.output_tokens)
        try:
            payload = _parse_json_payload(response.text)
        except ReplyParseError as exc:
            reason = str(exc)
            if response.stop_reason == "max_tokens":
                reason += " (truncated at max_tokens)"
            reasons.append(f"attempt {attempt}: {reason}")
            continue
        items = _items_from_payload(payload, chunk=chunk, split=split, licence_class=licence_class)
        return ChunkOutcome(status="ok", items=items, attempts=attempt, cost_usd=cost)
    return ChunkOutcome(
        status="skipped",
        items=[],
        attempts=MAX_ATTEMPTS,
        cost_usd=cost,
        reason="; ".join(reasons)[:300],
    )


def generate_items(
    chunks: Sequence[tuple[Chunk, str, str]],
    client: GuardedBedrockClient,
    *,
    dry_run: bool = False,
) -> list[Item] | DryRunEstimate:
    """In-memory generation (no files). chunks entries are (chunk, split, licence_class)."""
    plain_chunks = [entry[0] for entry in chunks]
    if dry_run:
        return project_dry_run(plain_chunks, guard=client.guard)
    out: list[Item] = []
    for chunk, split, licence_class in chunks:
        out.extend(generate_for_chunk(chunk, split, licence_class, client).items)
    return out


# --------------------------------------------------------------------------- resumable run


@dataclass
class GenerationSummary:
    chunks_total: int = 0
    already_done: int = 0
    chunks_ok: int = 0
    retried: int = 0
    skipped: int = 0
    items_generated: int = 0
    items_in_file: int = 0
    spend_this_run_usd: Decimal = Decimal("0")
    generation_spent_total_usd: Decimal = Decimal("0")
    generation_cap_usd: Decimal = Decimal("0")
    stopped: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "already_done": self.already_done,
            "chunks_ok": self.chunks_ok,
            "retried": self.retried,
            "skipped": self.skipped,
            "remaining": self.chunks_total - self.already_done - self.chunks_ok - self.skipped,
            "items_generated": self.items_generated,
            "items_in_file": self.items_in_file,
            "spend_this_run_usd": float(self.spend_this_run_usd),
            "generation_spent_total_usd": float(self.generation_spent_total_usd),
            "generation_cap_usd": float(self.generation_cap_usd),
            "stopped": self.stopped or None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GenerationPaths:
    generated: Path
    progress: Path
    failures: Path

    @classmethod
    def in_dir(cls, private: Path) -> GenerationPaths:
        return cls(
            generated=private / "generated.jsonl",
            progress=private / "generate_progress.jsonl",
            failures=private / "generate_failures.jsonl",
        )


def _read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, ignoring a torn trailing line left by a crash."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _append_lines(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(line + "\n" for line in lines))
        handle.flush()
        os.fsync(handle.fileno())


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    """Latest progress record per chunk key."""
    latest: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl_tolerant(path):
        key = record.get("chunk_key")
        if isinstance(key, str):
            latest[key] = record
    return latest


def _reconcile_generated(paths: GenerationPaths, progress: dict[str, dict[str, Any]]) -> int:
    """Keep only items from chunks recorded as finished; returns items kept."""
    if not paths.generated.exists():
        return 0
    if not paths.progress.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = paths.generated.with_name(f"generated.pre-resume-{stamp}.jsonl")
        paths.generated.replace(backup)
        return 0
    valid_ids = {
        str(item_id)
        for record in progress.values()
        if record.get("status") == "ok"
        for item_id in record.get("item_ids", [])
    }
    raw_lines = paths.generated.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    seen: set[str] = set()
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        item_id = value.get("id") if isinstance(value, dict) else None
        if isinstance(item_id, str) and item_id in valid_ids and item_id not in seen:
            seen.add(item_id)
            kept.append(line)
    if len(kept) != len([line for line in raw_lines if line.strip()]):
        tmp = paths.generated.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8", newline="\n")
        tmp.replace(paths.generated)
    return len(kept)


def run_generation(
    entries: Sequence[tuple[Chunk, str, str]],
    client: GuardedBedrockClient,
    paths: GenerationPaths,
    *,
    retry_failed: bool = False,
    on_progress: Callable[[int, int, GenerationSummary], None] | None = None,
) -> GenerationSummary:
    """Resumable, append-per-chunk generation. Ctrl-C stops cleanly after saving."""
    paths.generated.parent.mkdir(parents=True, exist_ok=True)
    summary = GenerationSummary(
        chunks_total=len(entries),
        generation_cap_usd=client.guard.phase_caps.get("generation", Decimal("0")),
    )
    progress = load_progress(paths.progress)
    had_legacy = paths.generated.exists() and not paths.progress.exists()
    summary.items_in_file = _reconcile_generated(paths, progress)
    if had_legacy:
        summary.notes.append("moved an earlier generated.jsonl aside (no progress file)")
    done_statuses = {"ok"} if retry_failed else {"ok", "skipped"}
    todo: list[tuple[Chunk, str, str]] = []
    for entry in entries:
        record = progress.get(chunk_key(entry[0]))
        if record is not None and record.get("status") in done_statuses:
            summary.already_done += 1
        else:
            todo.append(entry)
    try:
        for position, (chunk, split, licence_class) in enumerate(todo, start=1):
            try:
                outcome = generate_for_chunk(chunk, split, licence_class, client)
            except BudgetExceeded as exc:
                summary.stopped = f"budget cap reached: {exc}"
                break
            summary.spend_this_run_usd += outcome.cost_usd
            if outcome.attempts > 1:
                summary.retried += 1
            now = datetime.now(UTC).isoformat()
            if outcome.status == "ok":
                _append_lines(paths.generated, [item.model_dump_json() for item in outcome.items])
                summary.chunks_ok += 1
                summary.items_generated += len(outcome.items)
                summary.items_in_file += len(outcome.items)
            else:
                summary.skipped += 1
                failure = {
                    "timestamp": now,
                    "chunk_key": chunk_key(chunk),
                    "source_id": chunk.source_id,
                    "section_id": chunk.section_id,
                    "chunk_index": chunk.chunk_index,
                    "attempts": outcome.attempts,
                    "reason": outcome.reason,
                }
                _append_lines(paths.failures, [json.dumps(failure, sort_keys=True)])
            record = {
                "timestamp": now,
                "chunk_key": chunk_key(chunk),
                "status": outcome.status,
                "attempts": outcome.attempts,
                "item_ids": [item.id for item in outcome.items],
                "cost_usd": float(outcome.cost_usd),
            }
            _append_lines(paths.progress, [json.dumps(record, sort_keys=True)])
            if on_progress is not None:
                on_progress(position, len(todo), summary)
    except KeyboardInterrupt:
        summary.stopped = "interrupted; finished chunks are saved, rerun to resume"
    finally:
        summary.generation_spent_total_usd = client.guard.spent("generation")
    return summary


def make_fake_client(guard: BudgetGuard, text: str = '{"items": []}') -> GuardedBedrockClient:
    return GuardedBedrockClient(guard, FakeTransport(text=text))


def default_chunks_path() -> Path:
    return repo_root() / "data" / "private" / "chunks.jsonl"
