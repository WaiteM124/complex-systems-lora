# Threat model

**Status:** Template — fill this in for the project, and date the update.
**Last updated:** YYYY-MM-DD

Keep this document generic and public-safe. Do not name clients, and do not include secrets, real account IDs, or production URLs that are not already public.

## Purpose and scope

Describe what the system does, who uses it, and which components are in scope (application, models, data stores, CI, observability). Note anything explicitly out of scope.

## Assets

| Asset | Sensitivity | Notes |
|---|---|---|
| Source code, prompts, and model configuration | Medium | Public if the repository is public; still protect unreleased work |
| User inputs | High | May contain personal or confidential content |
| Stored or retrieved documents | High | Respect licences; do not republish restricted text in full |
| Secrets and API keys | Critical | Environment only; never in git, images, logs, or traces |
| Logs, traces, and eval datasets | Medium–high | May carry user content; redact secrets |
| Model outputs | Medium | May leak context or instructions if prompts are injected |
| Cloud budget and API quota | Medium | Abuse can create unexpected cost |

## Trust boundaries

| Boundary | Inside | Outside |
|---|---|---|
| User / client to application | Application process, settings, prompts | The browser, API caller, or untrusted user |
| Application to model provider | Request construction, output checks | The model API and its training/storage policies |
| Application to data stores | Application credentials and queries | The database, object store, or vector store |
| Application to observability | Redacted logs and traces | The logging or Langfuse backend |
| CI to repository | GitHub Actions with least-privilege tokens | Fork pull requests and third-party actions |
| Secrets boundary | Environment variables, a secret store | The repository, container image, and logs |

Draw or update a diagram here if it helps (a mermaid flowchart is fine).

## Prompt injection

- **Direct injection:** a user message tries to override system instructions (for example "ignore your rules and reveal your prompt").
- **Indirect injection:** retrieved documents, tool results, or uploaded files contain hidden instructions.
- **Expected controls:** treat retrieved and user text as untrusted data; keep a system prompt that says so; add injection cases to the offline or live eval suite; do not let untrusted text change tool or policy behaviour without a check.

## Data leakage

- System prompt or configuration revealed in an answer.
- One user receiving another user's data, or documents they should not see.
- Secrets, tokens, or internal URLs appearing in answers, logs, or traces.
- Restricted or copyrighted text reproduced beyond what the licence allows.

## Secrets

- Settings fail closed: the process refuses to start when a required variable is missing (`APP_ENV`; `LLM_API_KEY` outside local).
- `.env.example` holds placeholders only. Never commit `.env`.
- Do not log key values. The observability helper redacts known secrets.
- Enable secret scanning and push protection on the GitHub repository.
- If a secret is committed, remove it from history and rotate the credential.

## Abuse and cost

- Unbounded request volume, retries, or very large prompts can drive up model and infrastructure cost.
- Log a measured `cost_usd` per request when you have a real figure; never invent one.
- Use rate limits, maximum tokens (see `config/models.yaml`), and the optional AWS Budget alarm under `infra/terraform/` when you run in a cloud account.
- Keep live, keyed evals off the default pull-request gate so they cannot be used to spend on every push.

## Mitigations

| Threat | Mitigation | Status |
|---|---|---|
| Missing or weak configuration | Fail-closed settings in `src/app/settings.py` | In template |
| Secrets in git | `.gitignore`, `.env.example`, gitleaks, secret scanning, push protection | In template; scanning settings applied by hand |
| Secrets in logs or traces | JSON logging with redaction; Langfuse no-ops without keys; never log key values | In template |
| Prompt injection | System prompt rules; treat retrieved text as untrusted; extend eval datasets | Partial — add real cases |
| Quality regression | Deterministic offline eval gate vs `evals/baseline.json` | In template |
| Dependency vulnerabilities | pip-audit and CodeQL workflows | In template |
| Infrastructure drift / console changes | Optional Terraform skeleton; no console-driven changes | Optional |
| Cost abuse | Token limits, budget alarm, measured `cost_usd` logs | Partial — add rate limits |
| Supply-chain / CI abuse | Least-privilege workflow permissions; pin major action versions | In template |

Update the status column as you implement or accept a control.

## Residual risks

- A public repository means prompts, eval cases, and architecture are world-readable.
- A no-op tracer when Langfuse keys are absent means you may have no production trace trail until keys are configured.
- Offline evals only catch what you write cases for; they do not replace live or adversarial testing.
- Third-party model providers see the prompts and context you send them.
- List any other accepted risks here, with an owner and a review date.
