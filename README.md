# AI project template

A GitHub template with Miles Waite's engineering standards for new projects. Every repository created from it starts with agent and contributor docs, Cursor rules, security and lint CI, an offline eval gate, observability defaults, packaging, and a minimal Python skeleton that is already green.

## Create a new project from this template

**In the browser:** open this repository on GitHub and click **Use this template** → **Create a new repository**. Choose a name and visibility, then create it.

**With the GitHub CLI:**

```bash
gh repo create NAME --template WaiteM124/ai-project-template --public --clone
cd NAME
```

Use `--private` instead of `--public` if the new project should not be public.

## First steps after creating a repository

1. Fill in the placeholders in `AGENTS.md`, starting with `<PROJECT_PURPOSE>`, and rename the package in `pyproject.toml` if you do not want `ai-project`. Fill in `docs/threat-model.md` for the new project.
2. Apply a **ruleset on the default branch** (`main`): require a pull request, require the `lint-and-test` and `eval-gate` status checks (add CodeQL, pip-audit, and Terraform once they have a successful run), and block force pushes.
3. Enable **secret scanning** and **push protection** in the repository's security settings.
4. Copy `.env.example` to `.env`, keep `APP_ENV=local`, and install tools:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   make install
   make lint && make typecheck && make test && make eval
   ```

5. When you add real model or metric logic, replace the example cases in `evals/datasets/` and update `evals/baseline.json`. Keep this gate deterministic and offline. Live or keyed evals should be manual dispatch only.
6. If you do not need AWS infrastructure, delete `infra/terraform/` and `.github/workflows/terraform.yml`. If you do, replace the backend placeholders and pass `budget_alert_email` from an uncommitted tfvars file or from CI.
7. After creating the repo, update the security advisory URL in `.github/ISSUE_TEMPLATE/config.yml` if it still points at this template.

GitHub does not copy branch rulesets, secret scanning, or push protection from a template automatically, so steps 2 and 3 are done by hand on each new repository.

## Releases

Record notable changes in `CHANGELOG.md`. Pushing a tag that matches `v*` (for example `git tag v0.2.0 && git push origin v0.2.0`) creates a GitHub Release with generated notes. Do not tag a release without Miles's go-ahead.

## What is included

| Path | Purpose |
|---|---|
| `AGENTS.md` | Project purpose, non-negotiable principles, layout, and how to run and test |
| `.cursor/rules/` | Core, Python, Terraform, and frontend Cursor rules |
| `.github/pull_request_template.md` | Summary, changes, tests, principles checklist, no-secrets check |
| `.github/CODEOWNERS` | Default reviewer (`@WaiteM124`) |
| `.github/ISSUE_TEMPLATE/` | Bug report, feature request, and security contact link |
| `CONTRIBUTING.md` | Branch naming, PR flow, local checks |
| `CHANGELOG.md` | Keep a Changelog file |
| `.github/dependabot.yml` | Weekly pip and GitHub Actions updates (npm is included, commented out) |
| `docs/decisions/` | ADR index, ADR template, ADR-001 (use this template), ADR-002 (platform defaults) |
| `docs/threat-model.md` | Threat-model template (assets, boundaries, injection, leakage, secrets, cost) |
| `.pre-commit-config.yaml` | Ruff lint + format, gitleaks, and standard file hooks |
| `.github/workflows/lint-and-test.yml` | Ruff, mypy, and pytest via Make on every pull request and push to `main` |
| `.github/workflows/eval-gate.yml` | Deterministic offline eval against `evals/baseline.json`; fails on regression |
| `.github/workflows/codeql.yml` | CodeQL analysis for Python |
| `.github/workflows/security.yml` | pip-audit of installed dependencies |
| `.github/workflows/terraform.yml` | `terraform fmt` and `validate` when `infra/**` changes |
| `.github/workflows/release.yml` | Tag-triggered GitHub Release (`v*`) |
| `src/app/`, `tests/`, `evals/` | Python skeleton, fail-closed settings, observability, and eval datasets |
| `config/models.yaml` | Model ids and generation parameters per environment, loaded by code |
| `prompts/` | Versioned prompt files |
| `infra/terraform/` | Optional AWS skeleton (S3 state placeholders, default tags, budget alarm) |
| `Dockerfile`, `.dockerignore`, `Makefile` | Multi-stage non-root image and common developer targets |
| `.devcontainer/devcontainer.json` | Python 3.12 devcontainer that installs dev deps and pre-commit |
| `.env.example` | Placeholder-only environment variables that match `src/app/settings.py` |
| `.gitignore`, `SECURITY.md`, `LICENSE` | Ignore rules, vulnerability reporting, MIT licence |

## Working rules, in practice

- Branch from `main`, open a pull request, and wait for CI to go green. Do not push straight to `main`.
- Never commit secrets or real account IDs. Never invent metrics. Never use client code or data.
- The process fails closed without `APP_ENV`. Langfuse tracing no-ops when keys are absent and never logs key values.
- The initial commit that created this template repository itself was pushed directly to `main` because the repo was new and empty. That exception does not apply afterwards, and it does not apply to repositories created from the template.

## Licence

[MIT](LICENSE) © 2026 Miles Waite
