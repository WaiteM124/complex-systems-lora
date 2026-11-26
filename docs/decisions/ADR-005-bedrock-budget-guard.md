# ADR-005: Bedrock budget guard

**Status:** Accepted
**Date:** 2026-11-26

---

## Context

Generation and judging use Bedrock models with non-zero token prices. Unbounded calls could exceed the project's hard spend ceiling.

## Decision

- Caps live in `config/budget.yaml`: total `$4.50`, generation `$1.50`, eval `$3.00`.
- Every model call goes through `GuardedBedrockClient`, which requires a `BudgetGuard`.
- Before a call, reserve worst-case cost (estimated input + `max_tokens` output) using verified prices from `config/prices.yaml`. Missing or unverified prices are refused.
- After a call, append measured usage to `data/ledger/bedrock_spend.jsonl`.
- `BEDROCK_BUDGET_USD` may only **lower** the total cap.
- `--dry-run` projects cost without calling Bedrock. Cloud agents and CI use `FakeTransport` only.

## Consequences

- Live generation runs on Miles's machine with his AWS profile after merge.
- Tests assert the transport is not called when a reservation would exceed a cap.
