# ADR-002: Platform defaults for security, observability, and delivery

**Status:** Accepted
**Date:** 2026-11-26

---

## Context

The first version of this template covered contributor standards, PR flow, lint/test CI, and an offline eval gate. New projects still had to invent security scanning, a threat model, logging and tracing conventions, packaging, infrastructure, prompt and model configuration, fail-closed settings, and a release process. Those were easy to skip or to implement inconsistently.

## Decision

The template also ships:

- CodeQL (Python) and pip-audit workflows, with least-privilege permissions.
- A threat-model document to fill in per project.
- Structured JSON logging, a `cost_usd` field convention for measured request cost, and an optional Langfuse tracer that no-ops when keys are absent and never logs key values.
- A multi-stage non-root Dockerfile, `.dockerignore`, Makefile, and a Python devcontainer.
- CODEOWNERS, issue templates, a Keep a Changelog file, and a tag-triggered GitHub release workflow.
- An optional Terraform skeleton under `infra/terraform/` (S3 remote state placeholders, default tags, a budget alarm) and path-filtered Terraform CI. Delete the folder if unused.
- Versioned prompts, a YAML model config loaded by code, fail-closed pydantic-settings, and a JSONL eval dataset convention.

These remain generic. Repositories created from the template still fill in placeholders and apply branch rulesets, secret scanning, and push protection by hand.

## Alternatives

- Leave these as a documentation checklist only. That keeps the template smaller, but each new repo would re-implement them, and some would be missed.
- Make Terraform, Langfuse, and Docker mandatory. That would fight projects that do not need them. They stay optional (delete the Terraform tree; Langfuse no-ops without keys; Docker is unused until you build).

## Consequences

- New projects start with a broader, consistent baseline and greener defaults for security and operations.
- The template has more files to maintain (workflows, Terraform, Docker). Dependabot covers Actions and pip.
- CI gains jobs (CodeQL, pip-audit, Terraform on `infra/**`). Required status checks should be added once those workflows have a recorded successful run.
- Contributors must set `APP_ENV` (see `.env.example`) or the process will refuse to start.
