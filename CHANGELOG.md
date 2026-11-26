# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `generate` no longer aborts on one malformed model reply. Replies are repaired (unescaped inner quotes, LaTeX backslashes, fences, truncation at `max_tokens`); an unparseable reply is retried once, then the chunk is skipped and logged to `data/private/generate_failures.jsonl`.
- `generate` is resumable: items are appended per chunk with a progress file, so Ctrl-C or a crash keeps finished work and a rerun skips finished chunks (`--retry-failed` re-attempts skipped ones). Prints an end summary (ok, retried, skipped, items, spend).
- Budget caps hold across retries and reruns (spend is read from the persisted ledger before every call); reaching the cap stops cleanly instead of crashing. Reservations use a conservative input-token estimate.
- Item ids, split, licence class and provenance are always set by the pipeline, never taken from the model reply (model-supplied ids collided across chunks).
- JATS sources (cs-018) are parsed with the XML parser, so `XMLParsedAsHTMLWarning` is gone; section titles become headings and TeX maths is kept.
- `filter` skips a torn trailing line in `generated.jsonl`.

### Changed

- Generation prompt `generate_items.v2` asks for strict JSON with escaped quotes and backslashes. `max_tokens` stays 1200.

- Fetch validates Content-Type and magic bytes before saving; HTML landing pages are no longer written as PDFs.
- cs-041 points at the Think Complexity 2e PDF; cs-021 fetches Network Science HTML chapters; cs-018 uses the PLOS JATS manuscript URL.
- Chunk skips a bad source with a warning instead of crashing; exits non-zero only when nothing was chunked.
- Fetch prints a per-source summary table (ok / skipped / failed) covering all 46 IDs.
- arXiv e-prints that arrive as gzip/tar are unpacked to the main `.tex` before save.

### Added

- Milestone 1 data pipeline: licence-tiered sources, extract/chunk/split, budget-guarded Bedrock generation client, filters, leakage checks, notes loader, review CSV import/export, and CLI (`python -m app`).
- Config: `prices.yaml`, `budget.yaml`, `sources.yaml`, `concepts.yaml`, `data.yaml`; models updated for Qwen3-4B base, Qwen3 235B generator, DeepSeek V3.2 judge.
- Offline eval gate validates the Item schema and quote faithfulness on a synthetic tier-A fixture.
- ADRs 003–005 (licence tiers, free-GPU QLoRA, Bedrock budget guard); threat model covers licensing, private leakage, and budget overrun.

### Added (from template)

- CodeQL (Python) and pip-audit security workflows.
- Threat-model template, structured JSON logging, optional Langfuse tracing, and a measured `cost_usd` request-log convention.
- Fail-closed pydantic settings, versioned prompts, YAML model config, and a JSONL eval dataset convention.
- Multi-stage non-root Dockerfile, Makefile, devcontainer, and an optional Terraform skeleton with path-filtered CI.
- CODEOWNERS, issue templates, and a tag-triggered GitHub release workflow.

## [0.1.0] - 2026-11-26

### Added

- Initial engineering-standards template: AGENTS.md, Cursor rules, PR template, lint/test CI, offline eval gate, and a minimal Python skeleton.
