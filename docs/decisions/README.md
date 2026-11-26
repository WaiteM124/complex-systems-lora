# Architecture Decision Records

Lightweight ADRs for durable choices in this repository. Prefer reconstructing decisions from code and the README over inventing rationale.

## Format

Each record is `ADR-NNN-short-title.md`. Start new records from [ADR-000-template.md](ADR-000-template.md). Take the next free number, and rebase before adding one to avoid number clashes.

## Index

| ADR | Title | Status |
|---|---|---|
| [000](ADR-000-template.md) | ADR template | — |
| [001](ADR-001-use-ai-project-template.md) | Use the ai-project-template engineering standards | Accepted |
| [002](ADR-002-template-platform-defaults.md) | Platform defaults for security, observability, and delivery | Accepted |
| [003](ADR-003-licence-tiers.md) | Licence tiers for ingest and publication | Accepted |
| [004](ADR-004-free-gpu-qlora.md) | Free-GPU QLoRA training path | Accepted |
| [005](ADR-005-bedrock-budget-guard.md) | Bedrock budget guard | Accepted |

When behaviour changes, update the relevant ADR (or supersede it) and keep `AGENTS.md` in sync.
