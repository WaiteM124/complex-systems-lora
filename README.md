# complex-systems-lora

Licence-checked complex-systems grounded-QA data pipeline, free-GPU QLoRA fine-tune of Qwen3-4B, and an eval harness that gates the adapter against its base model. Portfolio project by Miles Waite. No client code or data.

## Licence tiers

| Tier | Sources | Train | Publish |
|---|---|---|---|
| **A** (CC BY) | cs-018, cs-033, cs-044 | Yes | Yes (with attribution) |
| **B** (CC BY-NC-SA) | cs-003, cs-021, cs-041 | Yes | **No** — `data/private/` only |
| **C** | All other `cs-0NN` | **Never** | Never |
| **Notes (N)** | Private module overviews | Optional | **No** |

Fixtures under `tests/fixtures/` are short synthetic CC0 text.

## Data pipeline (milestone 1)

1. **Fetch** raw sources into `data/private/raw/` (gitignored).
2. **Extract** text (LaTeX / HTML / PDF), keeping maths.
3. **Chunk** to ~300–500 tokens with section path metadata.
4. **Split by section** (80/10/10, seed `20261126`) before generation; commit test section IDs only.
5. **Generate** items with Qwen3 235B on Bedrock behind `GuardedBedrockClient` (over-generate ~1.4×). Resumable: each chunk is appended as it finishes, a rerun skips finished chunks, and a reply that is not valid JSON is repaired or retried once, then skipped and logged to `data/private/generate_failures.jsonl`.
6. **Filter** schema, verbatim quotes, concept vocabulary, near-duplicates, answerable ratio.
7. **Leakage check** — no test passage shares a 13-gram with train/val.
8. **Export review CSV** for hand-checking the test split; import fixes/drops.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env   # APP_ENV=local
make install
make lint && make typecheck && make test && make eval
make generate-dry      # projected cost; FakeTransport / fixtures; no AWS
make budget
```

## Commands after merge (Miles, on a machine with AWS)

1. `pip install -e ".[dev,data,bedrock]"`, then `python -m app fetch`
2. Optional: export module-overview notes as Markdown into `data/private/notes/`
3. `python -m app chunk && python -m app split && python -m app generate --dry-run` (worst case is informational; the live run checks the cap before every call)
4. `AWS_PROFILE=<profile> make generate` (safe to Ctrl-C and rerun; add `--retry-failed` to re-attempt skipped chunks), then `python -m app filter && python -m app leakage --drop-leaks`
5. `make review-export`, hand-check ~150 rows, then `make review-import`
6. Upload `reviewed.jsonl` (plus `filtered.jsonl` and `leakage_dropped.jsonl` for reference) to a **private** Kaggle dataset, then follow [Training on Kaggle](#training-on-kaggle-milestone-3)

## Task format

Passage + question in, strict JSON out (`prompts/answer_system.v1.txt` plus the concept vocabulary from `config/concepts.yaml`, built in `src/app/task.py`):

```json
{"answerable": true, "answer": "<= 3 sentences", "quote": "verbatim, <= 2 sentences", "concepts": ["1-3 tags"]}
```

Unanswerable questions return `answerable: false` with empty `answer`, `quote` and `concepts`. Each item in `reviewed.jsonl` carries its own `split` (`train`, `val`, `test`), fixed by the section-level split before generation; training refuses to start if any test item (by split, id, section or passage) would be trained on.

## Training on Kaggle (milestone 3)

Free T4 only (ADR-004). Recipe in `config/train.yaml`: QLoRA nf4, r=16, alpha=32, all linear projections, lr 2e-4, 3 epochs, max length 2048, batch 2 x accumulation 8, gradient checkpointing, fp16, one GPU. Loss is on the assistant JSON only. If CUDA runs out of memory it steps down to batch 1 x accumulation 16, then length 1536.

1. On Kaggle: **Create > New Notebook**, then **File > Import Notebook** and either paste the GitHub URL of `notebooks/train_qlora.ipynb` or upload the `.ipynb` file.
2. **Add Input** (right-hand panel): your private dataset `complex-systems-lora-data`. The notebook finds `reviewed.jsonl` anywhere under `/kaggle/input`.
3. **Settings > Accelerator: GPU T4 x2** (one GPU is used). **Settings > Internet: on**.
4. Optional, to publish the adapter: **Add-ons > Secrets**, add `HF_TOKEN` (a Hugging Face write token) and tick it for this notebook. Without it the adapter stays in `/kaggle/working/adapter`. The token is never printed.
5. **Run All**. The notebook installs the pinned `train` extra, checks the data split, runs a 20-step smoke test (prints peak VRAM), then the full run (prints runtime, peak VRAM and validation metrics). For a long run use **Save Version > Save & Run All** so it keeps going in the background.
6. Keep the notebook **private**: `/kaggle/working` holds checkpoints trained on private data.

Re-running resumes from the latest checkpoint in `/kaggle/working/checkpoints`. The best checkpoint is picked by validation answerability F1 on a fixed 48-item val subset. Same code without the notebook: `python -m app.train.qlora --config config/train.yaml [--smoke]` (add `--data path/to/reviewed.jsonl` off Kaggle; `make train-check` validates the data on a CPU).

## Evaluation (milestone 2)

Systems on the same hand-checked test split: (a) base model, (b) base + LoRA, (c) Qwen3 235B on Bedrock. Metrics (`src/app/scoring.py`): answerability F1 (macro over answerable/unanswerable; the primary gate), quote faithfulness (verbatim and <= 2 sentences), JSON validity, concept micro-F1, latency p50/p95 and tokens/sec labelled by hardware, and a DeepSeek V3.2 judge score (1-5, `prompts/judge_rubric.v1.txt`).

1. **Kaggle, (a) + (b):** import `notebooks/eval.ipynb` the same way, add the private dataset (and, in a new session, the training notebook's output or rely on the Hub adapter), GPU T4 x2, Internet on, **Run All**. Download `/kaggle/working/results/<run>.json` and commit it to `results/` in a PR.
2. **Your PC, (c) + judge** (own AWS login; always dry-run first). Copy the Kaggle `predictions/*.jsonl` into `data/private/predictions/` (gitignored, never commit), then:
   - `make baseline-dry`, then `AWS_PROFILE=<profile> make baseline`
   - `make judge-dry`, then `AWS_PROFILE=<profile> make judge`
   - `make score RUN=<run>` updates `results/<run>.json` with system (c) and the judge means.
3. `make eval` checks the newest run with base and LoRA: LoRA answerability F1 >= base + 10 points (quote faithfulness + 10 if base F1 > 85%), quote faithfulness >= 90%, JSON validity >= 98%, judge no more than 0.2 below base. It skips cleanly until results exist.

Both Bedrock steps use `GuardedBedrockClient` in the `eval` phase ($3.00 cap) and stop cleanly at the cap; rerun to resume.

## Budget

Hard caps in `config/budget.yaml`: total **$4.50**, generation **$1.50**, eval **$3.00**. `BEDROCK_BUDGET_USD` may only lower the total. Spend is read from the persisted ledger (`data/ledger/bedrock_spend.jsonl`) before every call, so the caps hold across retries and reruns. Prices are verified entries copied from `WaiteM124/rag-platform`.

## Layout

See [AGENTS.md](AGENTS.md) for principles, layout, and agent workflow. ADRs: licence tiers, free-GPU QLoRA, Bedrock budget guard.

## Licence

MIT (code). Corpus items follow their source licences; tier-B text is not published in this repository.
