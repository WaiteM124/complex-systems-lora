# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem, especially if it involves a leaked secret.

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/WaiteM124/ai-project-template/security/advisories/new) on this repository (or the repository created from this template). Include enough detail to reproduce the issue and any suggested fix.

This project is maintained by Miles Waite.

## Secrets

Never commit secrets, API keys, tokens, private keys, or real cloud account IDs. Use environment variables and `.env.example` with placeholders only.

If a secret is committed, remove it from the repository and its history, then rotate the exposed credentials.
