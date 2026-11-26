# AGENTS.md — contributor & agent guide

Single source of truth for humans and AI agents (including Cursor) working in this repository.

**Product:** `<PROJECT_PURPOSE>`
<!-- Replace the placeholder above with one short paragraph: what this project is and who it is for. -->

**Owner:** Miles Waite

**This repository is public unless you change its visibility.** Treat every commit as world-readable.

---

## Non-negotiable principles

1. **Every change goes through a pull request.** Do not commit or push straight to the default branch (`main`), including from a local machine. Open a branch, open a PR, and wait for CI to go green. The only exception is the initial commit that created a brand-new empty repository (see [ADR-001](docs/decisions/ADR-001-use-ai-project-template.md)).
2. **No secrets.** Never commit secrets, API keys, tokens, private keys, or real cloud account IDs. Use environment variables and `.env.example` with placeholders only. If a secret is found, remove it from git history and tell the owner which keys to rotate.
3. **Never invent metrics or results.** Report only what was actually run and measured. Do not fabricate latency, cost, quality scores, or eval numbers. The request log field `cost_usd` is only for a measured value.
4. **Never use client code or data.** Do not copy, ingest, or train on client material, and do not name client systems or projects in this repository.
5. **Tests are the spec.** Write tests first or alongside the code. Do not delete, skip, or weaken tests or eval baselines solely to make CI pass.
6. **Fail closed.** Security gates default to denied when configuration is missing. The process refuses to start if `APP_ENV` is unset, or if `LLM_API_KEY` is unset when `APP_ENV` is staging or production.
7. **Miles merges.** Agents do not merge pull requests, create releases, or dispatch privileged workflows without his go-ahead.

---

## Repository layout

| Path | Role |
|---|---|
| `src/app/` | Python package: settings, observability, model config, prompts, and application code |
| `tests/` | Unit and other automated tests (`pytest`) |
| `evals/` | Offline eval harness, baseline, and JSONL datasets under `evals/datasets/` |
| `config/models.yaml` | Model ids, temperature, and max tokens per environment (loaded by code) |
| `prompts/` | Versioned prompt files (`{name}.v{N}.txt`) |
| `infra/terraform/` | Optional AWS Terraform skeleton (delete if unused) |
| `docs/decisions/` | Architecture decision records (ADRs) |
| `docs/threat-model.md` | Threat-model template to fill in for this project |
| `.cursor/rules/` | Cursor rules that summarise this file for scoped globs |
| `.github/workflows/` | lint-and-test, eval-gate, CodeQL, pip-audit, Terraform (path-filtered), and release |

Stack: **Python 3.12+**. Add other languages or infra (frontend, Terraform) as the project needs them; rules for those areas already live under `.cursor/rules/`.

---

## How to run and test

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env   # then set APP_ENV=local; never commit .env
make install

make lint
make typecheck
make test
make eval
make run
```

`make docker-build` builds the multi-stage image when Docker is available. If you use the Terraform skeleton, run `terraform fmt -check` and `terraform validate` (after `terraform init -backend=false`) in `infra/terraform/`.

---

## Engineering standards

- Typed code. Type hints on public functions and methods.
- Small, focused PRs. One concern per PR; clear title and description; state assumptions and out-of-scope items.
- Pin versions for production-bound work. Do not silently bump models or critical dependencies in a feature PR. Change model ids in `config/models.yaml`, not in hardcoded Python strings.
- Do not log secret values. The observability helper redacts known secrets and no-ops Langfuse tracing when keys are absent.
- Docs track behaviour. Update the README, ADRs, CHANGELOG, and this file when behaviour, contracts, or run instructions change.
- When a Cursor rule and this file conflict, **this file wins**.

### Definition of done

- [ ] Principles above respected (PR flow, no secrets, honest metrics, no client material, fail closed)
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

---

## Related docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — branch naming, PR flow, and local checks
- [README.md](README.md) — how to use this template and what it includes
- [CHANGELOG.md](CHANGELOG.md) — notable changes
- [docs/decisions/](docs/decisions/) — architecture decision records
- [docs/threat-model.md](docs/threat-model.md) — threat-model template
- [SECURITY.md](SECURITY.md) — how to report vulnerabilities
