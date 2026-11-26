# complex-systems-lora

Licence-checked complex-systems grounded-QA data pipeline and (upcoming) free-GPU QLoRA fine-tune of Qwen3-4B. Portfolio project by Miles Waite. No client code or data.

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
6. Upload `data/private/` to a **private** Kaggle dataset (milestone 3)

## Budget

Hard caps in `config/budget.yaml`: total **$4.50**, generation **$1.50**, eval **$3.00**. `BEDROCK_BUDGET_USD` may only lower the total. Spend is read from the persisted ledger (`data/ledger/bedrock_spend.jsonl`) before every call, so the caps hold across retries and reruns. Prices are verified entries copied from `WaiteM124/rag-platform`.

## Layout

See [AGENTS.md](AGENTS.md) for principles, layout, and agent workflow. ADRs: licence tiers, free-GPU QLoRA, Bedrock budget guard.

## Licence

MIT (code). Corpus items follow their source licences; tier-B text is not published in this repository.
