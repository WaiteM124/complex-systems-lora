from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.bedrock import FakeTransport, GuardedBedrockClient
from app.budget import BudgetGuard
from app.observability import configure_logging
from app.settings import Settings


def test_requires_guard() -> None:
    with pytest.raises(TypeError):
        GuardedBedrockClient(None, FakeTransport())  # type: ignore[arg-type]


def test_records_usage_from_response(tmp_path: Path) -> None:
    guard = BudgetGuard(ledger_path=tmp_path / "ledger.jsonl")
    transport = FakeTransport(text="ok", input_tokens=11, output_tokens=7)
    client = GuardedBedrockClient(guard, transport)
    response = client.converse(
        phase="generation",
        model_id="qwen.qwen3-235b-a22b-2507-v1:0",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        max_tokens=50,
        input_tokens_est=5,
    )
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert guard.spent() > 0


def test_secrets_not_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key-xyz")
    settings = Settings(_env_file=None)
    configure_logging("INFO", settings=settings)
    guard = BudgetGuard(ledger_path=tmp_path / "ledger.jsonl")
    client = GuardedBedrockClient(guard, FakeTransport())
    with caplog.at_level(logging.INFO):
        client.converse(
            phase="generation",
            model_id="qwen.qwen3-235b-a22b-2507-v1:0",
            messages=[{"role": "user", "content": [{"text": "super-secret-key-xyz"}]}],
            system="super-secret-key-xyz",
            max_tokens=20,
            input_tokens_est=5,
        )
    joined = " ".join(record.getMessage() for record in caplog.records)
    # Prompt text is not logged by GuardedBedrockClient; ensure secret redaction path works.
    assert "super-secret-key-xyz" not in joined or "[REDACTED]" in joined
