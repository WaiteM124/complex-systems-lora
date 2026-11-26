# Eval datasets

Offline eval cases live here as JSONL. The eval gate (`evals/run_offline.py`) reads every `*.jsonl` file in this directory and scores them against `evals/baseline.json`. Keep this suite deterministic and free of network or API keys. Live or keyed evals belong in a manually dispatched workflow.

## Schema

Each line is one JSON object with these fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable unique id for the case |
| `input` | string | The input presented to the system |
| `expected` | string | The expected output (or golden answer) |
| `tags` | string array | Labels such as `smoke`, `uppercase`, or a topic |
| `source` | string | Where the case came from (for example `example`) |
| `licence` | string | Licence of the case text (for example `CC0-1.0`) |

Do not add secrets, client data, or copyrighted full text to a dataset.

## Adding cases

1. Append objects to an existing file, or add a new `*.jsonl` file.
2. Run `make eval` (or `python evals/run_offline.py`).
3. If the new cases should raise the bar, update `evals/baseline.json` in the same pull request and explain why. Do not lower the baseline just to make CI pass.
