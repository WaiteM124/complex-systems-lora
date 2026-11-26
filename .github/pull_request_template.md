## Summary

<!-- What changed and why. State assumptions and out-of-scope items. -->

## Changes

<!-- List the user-visible or contract changes. Link ADRs if a durable choice was made. -->

## Tests

<!-- Commands run locally, or "rely on CI". Name new or updated tests. -->

```bash
# ruff check . && ruff format --check . && mypy src && pytest -q && python evals/run_offline.py
```

## Checklist

- [ ] **Principles respected** — every change via PR; no invented metrics or results; no client code or data ([AGENTS.md](../AGENTS.md))
- [ ] **Tests added or updated** — existing tests and the eval baseline were not weakened or deleted to pass
- [ ] **Checks passing** — ruff, mypy, pytest, and the offline eval gate are green
- [ ] **No secrets** — no API keys, tokens, credentials, private keys, or real cloud account IDs
- [ ] **Docs updated** — README / ADRs / `AGENTS.md` / `CONTRIBUTING.md` updated if behaviour or workflow changed
