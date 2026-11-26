"""Budget guard for Bedrock spend. Fails closed on missing or unverified prices."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.paths import repo_root

TWOPLACE = Decimal("0.00000001")


class BudgetExceeded(RuntimeError):
    """Raised when a reservation would exceed a phase or total cap."""


class PriceError(RuntimeError):
    """Raised when a model price is missing or unverified."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    phase: str
    model_id: str
    input_tokens_est: int
    max_output_tokens: int
    worst_case_cost: Decimal


@dataclass(frozen=True)
class Price:
    input_per_1k: Decimal
    output_per_1k: Decimal
    verified: bool


def _money(value: Decimal | float | int | str) -> Decimal:
    return Decimal(str(value)).quantize(TWOPLACE, rounding=ROUND_HALF_UP)


@lru_cache(maxsize=1)
def load_budget_config(path: Path | None = None) -> dict[str, Any]:
    budget_path = path or (repo_root() / "config" / "budget.yaml")
    raw: Any = yaml.safe_load(budget_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("budget.yaml must be a mapping")
    return raw


@lru_cache(maxsize=1)
def load_prices(path: Path | None = None) -> dict[str, Price]:
    prices_path = path or (repo_root() / "config" / "prices.yaml")
    raw: Any = yaml.safe_load(prices_path.read_text(encoding="utf-8"))
    models = raw.get("models", {})
    out: dict[str, Price] = {}
    for model_id, entry in models.items():
        input_rate = entry.get("input_per_1k")
        output_rate = entry.get("output_per_1k")
        verified = bool(entry.get("verified", False))
        if input_rate is None or output_rate is None:
            out[model_id] = Price(
                input_per_1k=Decimal("0"),
                output_per_1k=Decimal("0"),
                verified=False,
            )
            continue
        out[model_id] = Price(
            input_per_1k=_money(input_rate),
            output_per_1k=_money(output_rate),
            verified=verified,
        )
    return out


def clear_budget_caches() -> None:
    load_budget_config.cache_clear()
    load_prices.cache_clear()


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: dict[str, Price] | None = None,
) -> Decimal:
    table = prices if prices is not None else load_prices()
    price = table.get(model_id)
    if price is None:
        raise PriceError(f"missing price for model {model_id!r}")
    if not price.verified:
        raise PriceError(f"unverified price for model {model_id!r}")
    cost = (Decimal(input_tokens) / Decimal(1000)) * price.input_per_1k + (
        Decimal(output_tokens) / Decimal(1000)
    ) * price.output_per_1k
    return _money(cost)


class BudgetGuard:
    """Reserves worst-case cost before a call and commits measured usage after."""

    def __init__(
        self,
        *,
        budget_path: Path | None = None,
        prices_path: Path | None = None,
        ledger_path: Path | None = None,
        total_cap_override: float | None = None,
        run_id: str | None = None,
    ) -> None:
        config = load_budget_config(budget_path)
        configured_total = _money(config["total_usd"])
        if total_cap_override is not None:
            override = _money(total_cap_override)
            if override > configured_total:
                raise ValueError("BEDROCK_BUDGET_USD may only lower the configured total cap")
            self.total_cap = override
        else:
            self.total_cap = configured_total
        phases_raw = config.get("phases", {})
        self.phase_caps: dict[str, Decimal] = {
            str(name): _money(value) for name, value in phases_raw.items()
        }
        self.prices = load_prices(prices_path)
        ledger_rel = str(config.get("ledger_path", "data/ledger/bedrock_spend.jsonl"))
        default_ledger = repo_root() / ledger_rel
        self.ledger_path = ledger_path or default_ledger
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            self.ledger_path.touch()
        self.run_id = run_id or uuid.uuid4().hex
        self._reserved: dict[str, Decimal] = {phase: Decimal("0") for phase in self.phase_caps}
        self._open: dict[str, Reservation] = {}

    def spent(self, phase: str | None = None) -> Decimal:
        total = Decimal("0")
        if not self.ledger_path.exists():
            return total
        for raw in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            entry = json.loads(line)
            if phase is not None and entry.get("phase") != phase:
                continue
            total += _money(entry["cost_usd"])
        return _money(total)

    def reserved(self, phase: str | None = None) -> Decimal:
        if phase is None:
            return _money(sum(self._reserved.values(), Decimal("0")))
        return _money(self._reserved.get(phase, Decimal("0")))

    def reserve(
        self,
        phase: str,
        model_id: str,
        input_tokens_est: int,
        max_output_tokens: int,
    ) -> Reservation:
        if phase not in self.phase_caps:
            raise BudgetExceeded(f"unknown budget phase {phase!r}")
        worst = estimate_cost(
            model_id,
            input_tokens_est,
            max_output_tokens,
            prices=self.prices,
        )
        phase_spent = self.spent(phase)
        total_spent = self.spent()
        phase_reserved = self.reserved(phase)
        total_reserved = self.reserved()
        if phase_spent + phase_reserved + worst > self.phase_caps[phase]:
            raise BudgetExceeded(
                f"phase {phase!r} cap exceeded: "
                f"spent={phase_spent} reserved={phase_reserved} worst={worst} "
                f"cap={self.phase_caps[phase]}"
            )
        if total_spent + total_reserved + worst > self.total_cap:
            raise BudgetExceeded(
                f"total cap exceeded: spent={total_spent} reserved={total_reserved} "
                f"worst={worst} cap={self.total_cap}"
            )
        reservation = Reservation(
            reservation_id=uuid.uuid4().hex,
            phase=phase,
            model_id=model_id,
            input_tokens_est=input_tokens_est,
            max_output_tokens=max_output_tokens,
            worst_case_cost=worst,
        )
        self._reserved[phase] = _money(self._reserved[phase] + worst)
        self._open[reservation.reservation_id] = reservation
        return reservation

    def commit(
        self,
        reservation: Reservation,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        open_res = self._open.pop(reservation.reservation_id, None)
        if open_res is None:
            raise RuntimeError(f"unknown reservation {reservation.reservation_id}")
        self._reserved[reservation.phase] = _money(
            max(Decimal("0"), self._reserved[reservation.phase] - reservation.worst_case_cost)
        )
        cost = estimate_cost(
            reservation.model_id,
            input_tokens,
            output_tokens,
            prices=self.prices,
        )
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "phase": reservation.phase,
            "model": reservation.model_id,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": float(cost),
            "run_id": self.run_id,
            "reservation_id": reservation.reservation_id,
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return cost

    def release(self, reservation: Reservation) -> None:
        open_res = self._open.pop(reservation.reservation_id, None)
        if open_res is None:
            return
        self._reserved[reservation.phase] = _money(
            max(Decimal("0"), self._reserved[reservation.phase] - reservation.worst_case_cost)
        )
