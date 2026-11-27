# ADR-004: Free-GPU QLoRA training path

**Status:** Accepted
**Date:** 2026-11-26

---

## Context

The portfolio goal is to fine-tune a small open-weight model without paid GPUs. The base model must fit a free 16 GB T4.

## Decision

- Base model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), configured in `config/models.yaml`.
- Training uses **QLoRA** (nf4) on free Kaggle T4 (Colab T4 as fallback). No paid GPU path in the main plan.
- Stack: Unsloth (Apache-2.0) + TRL `SFTTrainer` + PEFT, pinned in the `train` extra of `pyproject.toml` (never installed in CI). The Kaggle image's own torch build is kept.
- Recipe (`config/train.yaml`): nf4, r=16, alpha=32, all linear projections, lr 2e-4, 3 epochs, max length 2048, batch 2 x accumulation 8 (effective 16), gradient checkpointing, fp16 (the T4 has no bf16), one GPU of Kaggle's T4 x2. Loss on the assistant completion only.
- OOM fallback ladder: batch 1 x accumulation 16, then max length 1536. A 20-step smoke run logs peak VRAM and records the rung that fits.
- Checkpoints are saved to `/kaggle/working` and training resumes from the latest one; the adapter kept is the checkpoint with the best validation answerability F1 on a small fixed val subset (cheap greedy decoding).
- Entry points: `notebooks/train_qlora.ipynb` (thin, written fresh; Unsloth's LGPL-3.0 notebooks are not copied) and `python -m app.train.qlora`, which share one code path.
- Optional publication: the adapter is pushed to the HF Hub only when an `HF_TOKEN` env var or Kaggle Secret exists.

## Consequences

- Memory and batch settings target a single 16 GB T4 with fp16 (no bf16).
- Adapter weights may be published publicly with a model card that discloses training sources and the non-commercial note for tier B.
- Library pins are checked against PyPI for mutual compatibility but are only proven on Kaggle by the smoke run; bump them in a dedicated PR.
- Free-tier quotas and hardware can change without notice; resumable checkpoints and the Colab fallback limit the impact.
