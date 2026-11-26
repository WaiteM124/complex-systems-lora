from __future__ import annotations

import logging

import pytest
from pydantic import SecretStr

import app.observability as obs
from app.observability import get_tracer, log_request
from app.settings import Settings


def test_tracer_is_noop_when_keys_are_absent() -> None:
    settings = Settings(_env_file=None, app_env="local")
    tracer = get_tracer(settings)
    assert tracer.enabled is False
    with tracer.span("example"):
        pass


def test_request_log_redacts_secrets_and_omits_unmeasured_cost(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="local",
        llm_api_key=SecretStr("super-secret-test-key"),
    )
    obs._CONFIGURED = False
    obs.configure_logging("INFO", settings=settings)
    logging.getLogger("app").info("leaked %s", "super-secret-test-key")
    log_request(event="test", request_id="req-1", latency_ms=1.5)
    captured = capsys.readouterr().out
    assert "super-secret-test-key" not in captured
    assert "[REDACTED]" in captured
    assert "cost_usd" not in captured
    log_request(event="test", request_id="req-2", latency_ms=1.5, cost_usd=0.001)
    measured = capsys.readouterr().out
    assert "cost_usd" in measured
