# AGENTS.md — contributor & agent guide

Single source of truth for humans and AI agents (including Cursor) working in this repository.

**Product:** Licence-tiered complex-systems grounded-QA data pipeline and (later) free-GPU QLoRA fine-tune of Qwen3-4B, with budget-guarded Bedrock generation/eval and Helm serving. Portfolio project by Miles Waite; no client code or data.

**Owner:** Miles Waite

**This repository is public unless you change its visibility.** Treat every commit as world-readable.

---

## Non-negotiable principles

1. **Every change goes through a pull request.** Do not commit or push straight to the default branch (`main`), including from a local machine. Open a branch, open a PR, and wait for CI to go green. The only exception is the initial commit that created a brand-new empty repository (see [ADR-001](docs/decisions/ADR-001-use-ai-project-template.md)).
2. **No secrets.** Never commit secrets, API keys, tokens, private keys, or real cloud account IDs. Use environment variables and `.env.example` with placeholders only. If a secret is found, remove it from git history and tell the owner which keys to rotate.
3. **Never invent metrics or results.** Report only what was actually run and measured. Do not fabricate latency, cost, quality scores, or eval numbers. The request log field `cost_usd` is only for a measured value.
4. **Never use client code or data.** Do not copy, ingest, or train on client material, and do not name client systems or projects in this repository.
5. **Tests are the spec.** Write tests first or alongside the code. Do not delete, skip, or weaken tests or eval baselines solely to make CI pass.
6. **Fail closed.** Security gates default to denied when configuration is missing. The process refuses to start if `APP_ENV` is unset, or if `LLM_API_KEY` is unset when `APP_ENV` is staging or production. Bedrock calls without a `BudgetGuard` or with missing/unverified prices are refused.
7. **Miles merges.** Agents do not merge pull requests, create releases, or dispatch privileged workflows without his go-ahead.
8. **Licence tiers are law.** Tier A may be committed and published. Tier B may be trained on but never committed. Tier C is never ingested. Miles's notes are private (tier N) and never committed.
9. **Every Bedrock call is guarded.** Use `GuardedBedrockClient` only. CI and agents use `FakeTransport`; no live Bedrock or paid API calls in tests.
10. **Free-GPU-only training path.** QLoRA on free T4 GPUs (Kaggle primary, Colab fallback). No paid GPU path in the main plan.

---

## Repository layout

| Path | Role |
|---|---|
| `src/app/` | Python package: settings, observability, model config, prompts, budget guard, Bedrock client, schema, data pipeline, CLI |
| `src/app/data/` | Sources, extract, fetch, chunk, split, generate, filters, leakage, licence, notes, review import/export, reviewed-dataset loading |
| `src/app/train/` | QLoRA training (`python -m app.train.qlora`); GPU libraries imported lazily |
| `src/app/eval/`, `src/app/scoring.py` | Scorers, local GPU predictions, Bedrock baseline, judge, results files and the results gate |
| `notebooks/` | Thin Kaggle notebooks (`train_qlora.ipynb`, `eval.ipynb`); committed without outputs |
| `results/` | Aggregate-only eval results (`<run>.json`), no text |
| `tests/` | Unit and other automated tests (`pytest`), plus CC0 fixtures under `tests/fixtures/` |
| `evals/` | Offline eval harness, baseline, and JSONL datasets under `evals/datasets/` (tier A only) |
| `config/` | `models.yaml`, `prices.yaml`, `budget.yaml`, `sources.yaml`, `concepts.yaml`, `data.yaml`, `train.yaml`, `eval.yaml` |
| `prompts/` | Versioned prompt files (`{name}.v{N}.txt`) |
| `data/ledger/` | Committed Bedrock spend ledger (token counts and cost only) |
| `data/splits/` | Committed section-split IDs (no passage text) |
| `data/private/` | Gitignored private extracts, tier-B/N items, review CSV |
| `infra/terraform/` | Optional AWS Terraform skeleton (do not modify in M1) |
| `docs/decisions/` | Architecture decision records (ADRs) |
| `docs/threat-model.md` | Threat model (licensing, private leakage, budget) |
| `.cursor/rules/` | Cursor rules that summarise this file for scoped globs |
| `.github/workflows/` | lint-and-test, eval-gate, CodeQL, pip-audit, Terraform (path-filtered), and release |

Stack: **Python 3.12+**. Optional extras: `data` (extractors), `bedrock` (boto3), `train` (pinned GPU stack for Kaggle; never installed in CI, and tests must not import torch/unsloth/trl/peft at module import time).

---

## How to run and test

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env # then set APP_ENV=local; never commit .env
make install

make lint
make typecheck
make test
make eval
make generate-dry
make budget
```

`make docker-build` builds the multi-stage image when Docker is available. If you use the Terraform skeleton, run `terraform fmt -check` and `terraform validate` (after `terraform init -backend=false`) in `infra/terraform/`.

---

## Engineering standards

- Typed code. Type hints on public functions and methods.
- Small, focused PRs. One concern per PR; clear title and description; state assumptions and out-of-scope items.
- Pin versions for production-bound work. Do not silently bump models or critical dependencies in a feature PR. Change model ids in `config/models.yaml`, not in hardcoded Python strings. Prices live in `config/prices.yaml`.
- Do not log secret values or prompts. The observability helper redacts known secrets and no-ops Langfuse tracing when keys are absent. Log measured token usage and `cost_usd` only.
- Docs track behaviour. Update the README, ADRs, CHANGELOG, and this file when behaviour, contracts, or run instructions change.
- When a Cursor rule and this file conflict, **this file wins**.

### Definition of done

- [ ] Principles above respected (PR flow, no secrets, honest metrics, no client material, fail closed, licence tiers, budget guard)
- [ ] Tests added or updated; existing tests and eval baselines not weakened
- [ ] `make lint`, `make typecheck`, `make test`, and `make eval` pass
- [ ] Docs and CHANGELOG updated if behaviour or workflow changed

---

## Workflow for agents

1. **Read before writing.** Inspect the real layout, ADRs, neighbouring tests, and this file. Match existing patterns.
2. **Plan large changes.** For multi-file or cross-cutting work, outline the approach, files touched, and risks before coding. Keep scope to the assigned task.
3. **Do not weaken tests or eval baselines** solely to make CI pass.
4. **State assumptions in the PR.** Call out environment expectations, concurrent work, and follow-ups.
5. **Do not merge or tag a release** without Miles's go-ahead.
6. **Cursor rules.** Follow `.cursor/rules/*.mdc`. When in conflict, this file wins.
7. **No live paid APIs in CI or agent runs.** Use fixtures and `FakeTransport`.

---

## Related docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — branch naming, PR flow, and local checks
- [README.md](README.md) — purpose, licence tiers, pipeline steps, post-merge commands
- [CHANGELOG.md](CHANGELOG.md) — notable changes
- [docs/decisions/](docs/decisions/) — architecture decision records
- [docs/threat-model.md](docs/threat-model.md) — threat model
- [SECURITY.md](SECURITY.md) — how to report vulnerabilities
