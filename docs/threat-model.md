# Threat model

**Status:** Active for complex-systems-lora
**Last updated:** 2026-11-26

Keep this document generic and public-safe. Do not name clients, and do not include secrets, real account IDs, or production URLs that are not already public.

## Purpose and scope

Fine-tune a small open-weight model on licence-checked complex-systems material using free GPUs, evaluate it against a held-out set, and (later) serve it. In scope for milestone 1: the data pipeline, licence routing, leakage checks, and the Bedrock budget guard. Out of scope here: live Bedrock calls in CI, client systems, and paid GPU paths.

## Assets

| Asset | Sensitivity | Notes |
|---|---|---|
| Source code, prompts, and model configuration | Medium | Public repository |
| Tier-A public eval items | Low–medium | CC BY; attribution required |
| Tier-B / notes-derived private data | High | Never commit; `data/private/` gitignored |
| Secrets and API keys | Critical | Environment only |
| Bedrock spend ledger | Low | Token counts and cost only; no prompts |
| Cloud budget | Medium | Hard-capped in code |

## Trust boundaries

| Boundary | Inside | Outside |
|---|---|---|
| Operator to pipeline | Local CLI, settings, fixtures | Network fetches of raw sources |
| Application to Bedrock | `GuardedBedrockClient` + budget guard | AWS Bedrock API |
| Public repo to private data | Tracked files, eval fixtures | `data/private/`, `.env` |
| CI to repository | GitHub Actions, fake Bedrock client | Fork PRs, third-party actions |

## Data licensing

- Tier C sources are never ingested; tests refuse them.
- Tier B and notes (N) items must not appear in committed JSONL.
- `test_repo_clean` walks tracked files and fails on private paths, wrong licence classes, and stray tier-B/C `source_id` values outside allow-listed docs/config/tests.

## Private-data leakage

- `data/private/` is gitignored; CI fails if anything under it is tracked.
- Train/test leakage: no test passage may share a 13-gram with train or validation.
- Logs must not include prompts or secrets; observability redacts known secret values.

## Budget overrun

- Caps in `config/budget.yaml`; env override may only lower the total.
- Reserve-before-call semantics raise `BudgetExceeded` before the transport runs.
- Unverified or missing prices refuse the call.
- AWS account budget alerts (operator console / optional Terraform) are a backstop, not a substitute for the in-process guard.

## Prompt injection

Treat passage text as untrusted data. Generation and answer prompts require JSON that quotes from the passage; unanswerable near-misses exercise abstention.

## Secrets

- Fail-closed settings: `APP_ENV` required; `LLM_API_KEY` required outside local.
- `.env.example` holds placeholders only (`<ACCOUNT_ID>`).
- HF / Kaggle tokens are optional `SecretStr` fields and are included in redaction.

## Mitigations

| Threat | Mitigation | Status |
|---|---|---|
| Licence violation | Tier checks, routing, repo-clean CI | Implemented (M1) |
| Private data in git | gitignore + `test_repo_clean` | Implemented (M1) |
| Train/test leakage | 13-gram check | Implemented (M1) |
| Bedrock overspend | `BudgetGuard` + ledger | Implemented (M1) |
| Secrets in logs | Redaction helper | In template |
| Dependency risk | pip-audit, CodeQL | In template |
