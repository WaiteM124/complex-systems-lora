# ADR-001: Use the ai-project-template engineering standards

**Status:** Accepted
**Date:** 2026-11-26

---

## Context

New projects were starting as empty repositories, so agent and human contributors had to copy engineering standards by hand. That was easy to skip, and it left repos without a consistent PR flow, secret hygiene, Cursor rules, or CI gates.

## Decision

Every new project starts from the public GitHub template repository `WaiteM124/ai-project-template`. The template ships `AGENTS.md`, Cursor rules, a pull request template, `CONTRIBUTING.md`, Dependabot, ADRs, pre-commit hooks, lint/test CI, an offline eval gate, and a minimal Python skeleton that is green from day one.

The initial content of *this* template repository landed via pull request `template/initial-standards` into `main` (not a direct commit to `main`). After that first merge, and in every repository created from this template, all further changes go through a pull request.

## Alternatives

- Copy standards files from an existing project into each new repo by hand. This works, but it is easy to miss files or drift from the current set.
- A cookiecutter or other generator outside GitHub. That would also work, but it is an extra tool, and it would not appear as GitHub's "Use this template" button.

## Consequences

- New repos start with the same principles, PR template, and CI gates.
- Contributors still need to fill in `AGENTS.md` placeholders and apply repository settings (a default-branch ruleset, secret scanning, and push protection) after creating a repo from the template.
- Existing repositories are not changed automatically; they can adopt the files later if needed.
