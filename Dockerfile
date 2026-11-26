# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser prompts ./prompts
USER appuser
# Fail closed: pass APP_ENV at runtime, for example `docker run -e APP_ENV=local ...`
CMD ["python", "-m", "app"]
