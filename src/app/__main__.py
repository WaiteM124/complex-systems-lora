"""Load settings (fail closed) and configure logging. Used by `make run`."""

from __future__ import annotations

from app.observability import configure_logging, get_tracer, log_request
from app.settings import get_settings


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings=settings)
    get_tracer(settings)
    log_request(event="startup", request_id="startup", latency_ms=0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
