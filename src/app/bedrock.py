"""Budget-guarded Bedrock client. Live AWS calls require the optional bedrock extra."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.budget import BudgetGuard, Reservation
from app.observability import log_request


@dataclass(frozen=True)
class BedrockResponse:
    text: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, Any] | None = None


class BedrockTransport(Protocol):
    def converse(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> BedrockResponse: ...


class FakeTransport:
    """Deterministic transport for tests. No network."""

    def __init__(
        self,
        *,
        text: str = "{}",
        input_tokens: int = 10,
        output_tokens: int = 10,
    ) -> None:
        self.text = text
        self.input_tokens = input_tokens
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
        self.calls.append(
            {
                "model_id": model_id,
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return BedrockResponse(
            text=self.text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class Boto3ConverseTransport:
    """Live Bedrock Converse transport. Imports boto3 lazily."""

    def __init__(self, *, region: str) -> None:
        self.region = region

    def converse(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> BedrockResponse:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for live Bedrock calls; install the bedrock extra"
            ) from exc
        client = boto3.client("bedrock-runtime", region_name=self.region)
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]
        resp = client.converse(**kwargs)
        output = resp.get("output", {}).get("message", {}).get("content", [])
        text = "".join(part.get("text", "") for part in output)
        usage = resp.get("usage", {})
        return BedrockResponse(
            text=text,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            raw=resp if isinstance(resp, dict) else None,
        )


class GuardedBedrockClient:
    """The only supported way to call a model. Requires a BudgetGuard."""

    def __init__(self, guard: BudgetGuard, transport: BedrockTransport) -> None:
        if guard is None:
            raise TypeError("GuardedBedrockClient requires a BudgetGuard")
        if not isinstance(guard, BudgetGuard):
            raise TypeError("GuardedBedrockClient requires a BudgetGuard")
        self.guard = guard
        self.transport = transport

    def converse(
        self,
        *,
        phase: str,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int,
        temperature: float = 0.0,
        input_tokens_est: int | None = None,
    ) -> BedrockResponse:
        estimated_input = (
            input_tokens_est
            if input_tokens_est is not None
            else _estimate_input_tokens(messages, system)
        )
        reservation: Reservation = self.guard.reserve(
            phase,
            model_id,
            estimated_input,
            max_tokens,
        )
        started = time.perf_counter()
        try:
            response = self.transport.converse(
                model_id=model_id,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            self.guard.release(reservation)
            raise
        cost = self.guard.commit(
            reservation,
            response.input_tokens,
            response.output_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        log_request(
            event="bedrock_converse",
            request_id=reservation.reservation_id,
            latency_ms=latency_ms,
            cost_usd=float(cost),
            phase=phase,
            model_id=model_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return response


def _estimate_input_tokens(messages: list[dict[str, Any]], system: str | None) -> int:
    parts: list[str] = []
    if system:
        parts.append(system)
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
        else:
            parts.append(str(content))
    text = "\n".join(parts)
    return max(1, len(text.split()))
