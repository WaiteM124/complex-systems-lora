"""Filter generated items: schema, quote, tags, dedup, answerable ratio."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from pydantic import ValidationError

from app.data.chunk import load_data_config
from app.schema import Item, load_concept_vocabulary

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def ngrams(text: str, n: int = 5) -> set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_item(raw: dict[str, object]) -> Item | None:
    try:
        return Item.model_validate(raw)
    except ValidationError:
        return None


def quote_is_verbatim(item: Item) -> bool:
    quote = item.target.quote
    if not item.target.answerable:
        return quote == ""
    return bool(quote) and quote in item.passage


def concepts_in_vocabulary(item: Item) -> bool:
    vocabulary = load_concept_vocabulary()
    return all(tag in vocabulary for tag in item.target.concepts)


def deduplicate(items: Sequence[Item], *, jaccard_threshold: float = 0.9) -> list[Item]:
    kept: list[Item] = []
    seen_hashes: set[str] = set()
    kept_ngrams: list[set[str]] = []
    for item in items:
        key_text = f"{item.question}\n{item.target.answer}\n{item.target.quote}"
        digest = text_hash(key_text)
        if digest in seen_hashes:
            continue
        grams = ngrams(key_text, 5)
        if any(jaccard(grams, prev) >= jaccard_threshold for prev in kept_ngrams):
            continue
        seen_hashes.add(digest)
        kept_ngrams.append(grams)
        kept.append(item)
    return kept


def enforce_answerable_ratio(
    items: Sequence[Item],
    *,
    unanswerable_share: float | None = None,
) -> list[Item]:
    config = load_data_config()
    target_share = (
        unanswerable_share
        if unanswerable_share is not None
        else float(config.get("generation", {}).get("unanswerable_share", 0.2))
    )
    answerable = [item for item in items if item.target.answerable]
    unanswerable = [item for item in items if not item.target.answerable]
    if not items:
        return []
    # Keep all answerable; trim or keep unanswerable to approximate the share.
    if not answerable:
        return list(unanswerable)
    desired_unans = int(round(len(answerable) * target_share / max(1e-9, 1 - target_share)))
    kept_unans = unanswerable[:desired_unans]
    return answerable + kept_unans


def filter_items(items: Iterable[dict[str, object] | Item]) -> list[Item]:
    validated: list[Item] = []
    for raw in items:
        item = raw if isinstance(raw, Item) else validate_item(raw)
        if item is None:
            continue
        if not quote_is_verbatim(item):
            continue
        if not concepts_in_vocabulary(item):
            continue
        validated.append(item)
    deduped = deduplicate(validated)
    return enforce_answerable_ratio(deduped)
