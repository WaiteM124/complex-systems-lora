"""Structured JSON logging, optional tracing, and a cost-per-request convention.

Cost convention
---------------
Request logs and traces use the field name `cost_usd` for a measured cost in
US dollars. Only include it when the caller actually measured a value. Never
invent or estimate a cost to fill the field.

Langfuse
--------
`get_tracer` returns a no-op tracer when Langfuse keys are absent, the
optional `langfuse` extra is not installed, or the client fails to start.
Key values are never logged.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from app.settings import Settings

_COST_FIELD = "cost_usd"
_CONFIGURED = False
_SECRET_VALUES: list[str] = []

LOGGER = logging.getLogger("app")


def _redact(text: str) -> str:
    redacted = text
    for secret in _SECRET_VALUES:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    reserved = ("request_id", "event", "latency_ms", _COST_FIELD)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact(record.getMessage()),
        }
        for key in self.reserved:
            if hasattr(record, key):
                value = getattr(record, key)
                if key == _COST_FIELD and value is None:
                    continue
                payload[key] = value if not isinstance(value, str) else _redact(value)
        if record.exc_info:
            payload["exception"] = _redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, settings: Settings | None = None) -> None:
    """Idempotent JSON logging to stdout. Optional `settings` enable secret redaction."""
    global _CONFIGURED, _SECRET_VALUES
    if settings is not None:
        _SECRET_VALUES = settings.secret_values()
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def log_request(
    *,
    event: str,
    request_id: str,
    latency_ms: float,
    cost_usd: float | None = None,
    **extra: Any,
) -> None:
    """Log one request. Pass `cost_usd` only when it was measured."""
    payload: dict[str, Any] = {
        "event": event,
        "request_id": request_id,
        "latency_ms": latency_ms,
    }
    if cost_usd is not None:
        payload[_COST_FIELD] = cost_usd
    payload.update(extra)
    LOGGER.info("request %s", event, extra=payload)


class NoOpTracer:
    """Used when Langfuse is disabled or unavailable."""

    enabled: bool = False

    @contextmanager
    def span(self, name: str, **_attributes: Any) -> Iterator[None]:
        yield None


class LangfuseTracer:
    """Thin wrapper around the optional Langfuse client."""

    enabled: bool = True

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        start = getattr(self._client, "start_as_current_span", None) or getattr(
            self._client, "start_span", None
        )
        if start is None:
            yield None
            return
        ctx = start(name=name)
        try:
            span = ctx.__enter__() if hasattr(ctx, "__enter__") else ctx
            if attributes and span is not None:
                update = getattr(span, "update", None)
                if callable(update):
                    update(**attributes)
            yield span
        except Exception:
            if hasattr(ctx, "__exit__"):
                ctx.__exit__(*sys.exc_info())
            raise
        else:
            if hasattr(ctx, "__exit__"):
                ctx.__exit__(None, None, None)


def get_tracer(settings: Settings | None = None) -> NoOpTracer | LangfuseTracer:
    """Return a Langfuse tracer, or a no-op if keys are missing or setup fails."""
    if settings is None:
        return NoOpTracer()
    public = settings.langfuse_public_key
    secret = settings.langfuse_secret_key
    if public is None or secret is None:
        return NoOpTracer()
    public_value = public.get_secret_value()
    secret_value = secret.get_secret_value()
    if not public_value or not secret_value:
        return NoOpTracer()
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        LOGGER.info("Langfuse package not installed; tracing disabled")
        return NoOpTracer()
    kwargs: dict[str, Any] = {
        "public_key": public_value,
        "secret_key": secret_value,
    }
    if settings.langfuse_host:
        kwargs["host"] = settings.langfuse_host
    try:
        client = Langfuse(**kwargs)
    except Exception:
        LOGGER.warning("Langfuse client failed to start; tracing disabled")
        return NoOpTracer()
    LOGGER.info("Langfuse tracing enabled")
    return LangfuseTracer(client)
