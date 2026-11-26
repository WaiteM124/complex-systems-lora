# ADR-004: Free-GPU QLoRA training path

**Status:** Accepted
**Date:** 2026-11-26

---

## Context

The portfolio goal is to fine-tune a small open-weight model without paid GPUs. The base model must fit a free 16 GB T4.

## Decision

- Base model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), configured in `config/models.yaml`.
- Training uses **QLoRA** (nf4) on free Kaggle T4 (Colab T4 as fallback). No paid GPU path in the main plan.
- Training code lands in a later milestone; this ADR locks the free-GPU constraint for the project.

## Consequences

- Memory and batch settings target a single 16 GB T4 with fp16 (no bf16).
- Adapter weights may be published publicly with a model card that discloses training sources and the non-commercial note for tier B.
