# Optional Terraform skeleton

This directory is a starting point for AWS infrastructure. **Delete it** (and `.github/workflows/terraform.yml`) if the project does not need Terraform.

## What it contains

- An S3 remote-state backend block with placeholders (`<STATE_BUCKET>`, `<PROJECT>`, `<STATE_LOCK_TABLE>`).
- Provider `default_tags` for `project`, `owner`, `env`, and `managed-by`.
- An AWS Budgets monthly cost alarm that emails `var.budget_alert_email`.
- No real account IDs, buckets, or addresses. Use `<ACCOUNT_ID>` if you add an account reference.

## Before you apply

1. Create (or reuse) a state bucket and lock table in your account, then replace the backend placeholders. `terraform init` will fail until those values are real bucket and table names.
2. Pass `project`, `owner`, `environment`, and `budget_alert_email` via a `*.tfvars` file that is **not** committed (see `.gitignore`), or via CI variables.
3. Never commit secrets, real AWS account IDs, or production email addresses.
4. Do not make console-driven resource changes. All infrastructure stays in this tree.

CI runs `terraform fmt -check` and `terraform validate` (with `-backend=false`) only when `infra/**` changes.
