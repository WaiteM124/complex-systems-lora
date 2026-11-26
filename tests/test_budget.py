from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.bedrock import FakeTransport, GuardedBedrockClient
from app.budget import BudgetExceeded, BudgetGuard, PriceError, clear_budget_caches, estimate_cost
from app.data.chunk import Chunk
from app.data.generate import project_dry_run
from app.settings import Settings


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    clear_budget_caches()


def test_reservation_exceeds_phase_before_transport(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger)
    # Exhaust generation phase with a huge reservation estimate.
    transport = FakeTransport()
    client = GuardedBedrockClient(guard, transport)
    with pytest.raises(BudgetExceeded):
        client.converse(
            phase="generation",
            model_id="qwen.qwen3-235b-a22b-2507-v1:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            max_tokens=1200,
            input_tokens_est=10_000_000,
        )
    assert len(transport.calls) == 0


def test_total_cap_across_phases(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger, total_cap_override=0.001)
    # First tiny commit on generation.
    res = guard.reserve("generation", "qwen.qwen3-235b-a22b-2507-v1:0", 10, 10)
    guard.commit(res, 10, 10)
    with pytest.raises(BudgetExceeded):
        guard.reserve("eval", "deepseek.v3.2", 500_000, 400)


def test_missing_or_unverified_price_raises(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger)
    with pytest.raises(PriceError):
        guard.reserve("generation", "missing.model", 10, 10)


def test_bedrock_budget_usd_lower_ok_higher_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("BEDROCK_BUDGET_USD", "1.0")
    settings = Settings(_env_file=None)
    assert settings.bedrock_budget_usd == 1.0
    monkeypatch.setenv("BEDROCK_BUDGET_USD", "999.0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_ledger_cost_for_qwen_tokens(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger)
    model = "qwen.qwen3-235b-a22b-2507-v1:0"
    expected = estimate_cost(model, 1000, 1000)
    assert expected == Decimal("0.00171")  # 0.00034 + 0.00137
    res = guard.reserve("generation", model, 1000, 1000)
    cost = guard.commit(res, 1000, 1000)
    assert cost == expected
    assert guard.spent() == expected
    line = ledger.read_text(encoding="utf-8").strip()
    assert "qwen.qwen3-235b-a22b-2507-v1:0" in line


def test_dry_run_makes_zero_calls(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    guard = BudgetGuard(ledger_path=ledger)
    transport = FakeTransport()
    client = GuardedBedrockClient(guard, transport)
    chunks = [
        Chunk(
            section_path="Intro",
            section_id="fixture/intro",
            text="[Intro]\nA short synthetic passage about fixed points and stability.",
            token_count=10,
            chunk_hash="abc",
            chunk_index=0,
            source_id="fixture",
        )
    ]
    estimate = project_dry_run(chunks, guard=guard)
    assert estimate.call_count >= 1
    assert float(estimate.worst_case_cost_usd) >= 0.0
    assert len(transport.calls) == 0
    # Also ensure client was unused.
    assert client.transport is transport
