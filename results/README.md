# Results

One JSON file per evaluation run, `results/<run>.json`, written by
`python -m app.eval.score`. Files hold **aggregate metrics only**: no passages,
questions or model outputs (the schema in `src/app/eval/results.py` forbids extra
fields). Each file records the SHA-256 of `reviewed.jsonl` and of the canonical
test split, so runs on a different test set cannot be merged.

Systems: `base` (a), `lora` (b) and `qwen3-235b` (c, Bedrock). Rates are fractions
in [0, 1]; latency is labelled by hardware.

`make eval` validates every file here and gates the newest run that holds both
`base` and `lora` against `evals/baseline.json` (`gate`). Never edit numbers by
hand; re-run the scorer instead.
