# Contributing

Short human guide. Agents and non-negotiable principles: see [AGENTS.md](AGENTS.md).

## Setup

**Prerequisites:** Python 3.12+ and pip. Docker and Terraform are optional.

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env   # placeholders only; set APP_ENV=local; never commit .env
make install
```

## Local checks

Run these before opening a pull request:

```bash
make lint
make typecheck
make test
make eval
```

If you change `infra/**`:

```bash
cd infra/terraform
terraform init -backend=false
terraform fmt -check
terraform validate
```

Do not lower the committed baseline in `evals/baseline.json` just to make CI pass.

## Branch naming

Branch from `main` with a short, descriptive name:

| Prefix | Use for |
|---|---|
| `feat/<short-slug>` | New behaviour |
| `fix/<short-slug>` | Bug fixes |
| `docs/<short-slug>` | Documentation only |
| `chore/<short-slug>` | Tooling, CI, or dependency housekeeping |
| `refactor/<short-slug>` | Behaviour-preserving restructuring |

Examples: `feat/offline-eval-cases`, `fix/mypy-src-layout`.

## Pull request flow

1. Create a branch from an up-to-date `main`.
2. Make a small, focused change (one concern per PR).
3. Run the local checks above.
4. Open a pull request against `main` and fill in the pull request template.
5. Wait for **Lint and test** and **Eval gate** to go green (and for Security, CodeQL, or Terraform when those workflows run).
6. Do not merge, tag a release, or trigger privileged workflows without Miles's go-ahead. Do not push straight to `main`.

State assumptions, concurrent work, and follow-ups in the description. Do not delete or weaken tests or eval baselines solely to make CI pass.

## Definition of done

- Principles in `AGENTS.md` respected (PR flow, no secrets, honest metrics, no client material, fail closed)
- Tests added or updated and passing; lint and type checks clean; eval gate green
- No secrets, keys, or real cloud account IDs in the change
- Docs and CHANGELOG updated when behaviour or workflow changes
- Existing CI green on the PR
