"""Pure scorers for the grounded-QA task. No I/O, no model calls.

Definitions (all rates are fractions in [0, 1]):

- **JSON validity**: the raw model output, stripped of surrounding whitespace, is a
  single JSON object with exactly-typed `answerable` (bool), `answer` (str),
  `quote` (str) and `concepts` (list of str). Code fences or extra prose count as
  invalid.
- **Answerability F1**: macro-averaged F1 over the two classes (answerable,
  unanswerable). Outputs are parsed leniently for this (fences and prose around
  one JSON object are tolerated); an output with no parseable object counts as a
  wrong prediction. Macro averaging stops an always-answer model scoring well on
  the ~80/20 class mix. Per-class F1 is reported too.
- **Quote faithfulness**: over outputs that claim `answerable: true`, the share
  whose quote is non-empty, a verbatim substring of the passage, and at most two
  sentences.
- **Concept micro-F1**: tag sets compared per item, TP/FP/FN summed over items.
- **Latency** p50/p95 (ms, linear interpolation) and **tokens/sec** (total output
  tokens divided by total generation seconds).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.schema import count_sentences

REQUIRED_KEYS = ("answerable", "answer", "quote", "concepts")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedOutput:
    strict_valid: bool
    parsed: bool
    answerable: bool | None = None
    answer: str = ""
    quote: str = ""
    concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredExample:
    """One gold item paired with one system output."""

    gold_answerable: bool
    gold_concepts: tuple[str, ...]
    passage: str
    output: str
    latency_ms: float | None = None
    output_tokens: int | None = None


def _typed_target(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if any(key not in obj for key in REQUIRED_KEYS):
        return None
    if not isinstance(obj["answerable"], bool):
        return None
    if not isinstance(obj["answer"], str) or not isinstance(obj["quote"], str):
        return None
    concepts = obj["concepts"]
    if not isinstance(concepts, list) or not all(isinstance(c, str) for c in concepts):
        return None
    return obj


def _strict(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return _typed_target(obj)


def _lenient(text: str) -> dict[str, Any] | None:
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    found = _strict(cleaned)
    if found is not None:
        return found
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    return _strict(cleaned[start : end + 1])


def parse_output(text: str) -> ParsedOutput:
    strict = _strict(text)
    obj = strict if strict is not None else _lenient(text)
    if obj is None:
        return ParsedOutput(strict_valid=False, parsed=False)
    return ParsedOutput(
        strict_valid=strict is not None,
        parsed=True,
        answerable=bool(obj["answerable"]),
        answer=str(obj["answer"]),
        quote=str(obj["quote"]),
        concepts=tuple(str(c) for c in obj["concepts"]),
    )


def json_validity(outputs: Sequence[str]) -> float | None:
    if not outputs:
        return None
    return sum(1 for text in outputs if _strict(text) is not None) / len(outputs)


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def answerability_f1(
    gold: Sequence[bool],
    predicted: Sequence[bool | None],
) -> dict[str, float | None]:
    """Macro F1 over answerable/unanswerable. `None` predictions count as wrong."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    if not gold:
        return {"macro": None, "answerable": None, "unanswerable": None}
    resolved = [(not g) if p is None else p for g, p in zip(gold, predicted, strict=True)]
    per_class: dict[str, float] = {}
    for label, name in ((True, "answerable"), (False, "unanswerable")):
        tp = sum(1 for g, p in zip(gold, resolved, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, resolved, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, resolved, strict=True) if g == label and p != label)
        per_class[name] = _f1(tp, fp, fn)
    macro = (per_class["answerable"] + per_class["unanswerable"]) / 2
    return {
        "macro": macro,
        "answerable": per_class["answerable"],
        "unanswerable": per_class["unanswerable"],
    }


def quote_is_faithful(quote: str, passage: str) -> bool:
    return bool(quote.strip()) and quote in passage and count_sentences(quote) <= 2


def quote_faithfulness(pairs: Iterable[tuple[ParsedOutput, str]]) -> tuple[float | None, int]:
    """Share of answerable-claiming outputs with a faithful quote, and that count."""
    claimed = [(p, passage) for p, passage in pairs if p.parsed and p.answerable]
    if not claimed:
        return None, 0
    ok = sum(1 for p, passage in claimed if quote_is_faithful(p.quote, passage))
    return ok / len(claimed), len(claimed)


def concept_micro_f1(
    gold: Sequence[Iterable[str]],
    predicted: Sequence[Iterable[str]],
) -> float | None:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    tp = fp = fn = 0
    for g_raw, p_raw in zip(gold, predicted, strict=True):
        g, p = set(g_raw), set(p_raw)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    if tp + fp + fn == 0:
        return None
    return _f1(tp, fp, fn)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def tokens_per_second(
    latencies_ms: Sequence[float],
    output_tokens: Sequence[int],
) -> float | None:
    if len(latencies_ms) != len(output_tokens):
        raise ValueError("latency and token lengths differ")
    total_s = sum(latencies_ms) / 1000.0
    if total_s <= 0:
        return None
    return sum(output_tokens) / total_s


def _round(value: float | None, ndigits: int = 4) -> float | None:
    return None if value is None else round(value, ndigits)


def score_examples(examples: Sequence[ScoredExample]) -> dict[str, float | int | None]:
    """Aggregate metrics for one system. Returns numbers only, never text."""
    parsed = [parse_output(ex.output) for ex in examples]
    f1 = answerability_f1(
        [ex.gold_answerable for ex in examples],
        [p.answerable if p.parsed else None for p in parsed],
    )
    faith, n_claimed = quote_faithfulness(
        (p, ex.passage) for p, ex in zip(parsed, examples, strict=True)
    )
    concepts = concept_micro_f1(
        [ex.gold_concepts for ex in examples],
        [p.concepts if p.parsed else () for p in parsed],
    )
    timed = [
        (ex.latency_ms, ex.output_tokens)
        for ex in examples
        if ex.latency_ms is not None and ex.output_tokens is not None
    ]
    latencies = [float(lat) for lat, _ in timed if lat is not None]
    tokens = [int(tok) for _, tok in timed if tok is not None]
    return {
        "n_items": len(examples),
        "n_gold_answerable": sum(1 for ex in examples if ex.gold_answerable),
        "answerability_f1": _round(f1["macro"]),
        "answerable_class_f1": _round(f1["answerable"]),
        "unanswerable_class_f1": _round(f1["unanswerable"]),
        "quote_faithfulness": _round(faith),
        "n_quotes_scored": n_claimed,
        "json_validity": _round(json_validity([ex.output for ex in examples])),
        "concept_micro_f1": _round(concepts),
        "latency_p50_ms": _round(percentile(latencies, 50), 1),
        "latency_p95_ms": _round(percentile(latencies, 95), 1),
        "tokens_per_sec": _round(tokens_per_second(latencies, tokens), 2) if timed else None,
    }
