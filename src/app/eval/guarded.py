"""Shared helpers for budget-guarded Bedrock eval calls (phase `eval`).

Live calls only ever run on the owner's machine with his own AWS login. CI and
agents use `FakeTransport`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.bedrock import (
    BedrockTransport,
    Boto3ConverseTransport,
    FakeTransport,
    GuardedBedrockClient,
)
from app.budget import BudgetGuard, estimate_cost
from app.data.generate import reservation_input_tokens
from app.model_config import get_region

EVAL_PHASE = "eval"


@dataclass(frozen=True)
class Call:
    """One planned Bedrock call: an id, a system prompt and a user prompt."""

    id: str
    system: str
    user: str


@dataclass(frozen=True)
class Projection:
    calls: int
    worst_case_usd: Decimal
    eval_cap_usd: Decimal
    eval_spent_usd: Decimal
    total_cap_usd: Decimal
    total_spent_usd: Decimal

    @property
    def eval_remaining_usd(self) -> Decimal:
        return self.eval_cap_usd - self.eval_spent_usd

    def as_dict(self, *, model_id: str, max_tokens: int) -> dict[str, Any]:
        return {
            "dry_run": True,
            "phase": EVAL_PHASE,
            "model_id": model_id,
            "calls": self.calls,
            "max_tokens": max_tokens,
            "worst_case_cost_usd": float(self.worst_case_usd),
            "eval_cap_usd": float(self.eval_cap_usd),
            "eval_spent_usd": float(self.eval_spent_usd),
            "eval_remaining_usd": float(self.eval_remaining_usd),
            "total_cap_usd": float(self.total_cap_usd),
            "total_spent_usd": float(self.total_spent_usd),
            "fits_remaining_eval_budget": self.worst_case_usd <= self.eval_remaining_usd,
            "note": (
                "worst case assumes every call uses max_tokens and a conservative input "
                "estimate; the live run checks the cap before every call and stops "
                "cleanly (resumable) if the next call could exceed it"
            ),
        }


def project(
    calls: Sequence[Call], *, guard: BudgetGuard, model_id: str, max_tokens: int
) -> Projection:
    worst = sum(
        (
            estimate_cost(
                model_id,
                reservation_input_tokens(call.system, call.user),
                max_tokens,
                prices=guard.prices,
            )
            for call in calls
        ),
        Decimal("0"),
    )
    return Projection(
        calls=len(calls),
        worst_case_usd=worst,
        eval_cap_usd=guard.phase_caps.get(EVAL_PHASE, Decimal("0")),
        eval_spent_usd=guard.spent(EVAL_PHASE),
        total_cap_usd=guard.total_cap,
        total_spent_usd=guard.spent(),
    )


def make_client(
    guard: BudgetGuard,
    *,
    fake: bool,
    fake_text: str,
    region: str | None = None,
) -> GuardedBedrockClient:
    transport: BedrockTransport
    if fake:
        transport = FakeTransport(text=fake_text)
    else:
        transport = Boto3ConverseTransport(region=region or get_region())
    return GuardedBedrockClient(guard, transport)
