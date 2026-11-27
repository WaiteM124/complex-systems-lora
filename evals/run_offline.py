#!/usr/bin/env python3
"""Deterministic offline eval gate. No network calls, no keys.

1. Tier-A slice: validates every JSONL case in evals/datasets/ against the Item
   schema, requires licence_class == "A", and checks quote faithfulness against
   evals/baseline.json `metrics`.
2. Results gate: validates every committed results/*.json (aggregates only) and
   checks the newest run holding both base and LoRA against evals/baseline.json
   `gate` (see src/app/eval/gate.py). Skips cleanly when no results exist yet.

Exits non-zero on any regression or failed gate check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.eval.gate import GateConfig, run_gate  # noqa: E402
from app.schema import Item  # noqa: E402

BASELINE_PATH = ROOT / "evals" / "baseline.json"
DATASETS_DIR = ROOT / "evals" / "datasets"


def load_items(directory: Path = DATASETS_DIR) -> list[Item]:
    items: list[Item] = []
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL datasets in {directory}")
    for path in paths:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = Item.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 — gate reports path/line
                raise ValueError(f"{path.name}:{line_no} schema invalid: {exc}") from exc
            if item.licence_class != "A":
                raise ValueError(
                    f"{path.name}:{line_no} licence_class must be A, got {item.licence_class}"
                )
            items.append(item)
    return items


def score(items: list[Item]) -> dict[str, float]:
    if not items:
        raise ValueError("no eval cases")
    schema_ok = 1.0  # load_items already validated
    faithful = 0
    for item in items:
        quote = item.target.quote
        if not item.target.answerable:
            if quote == "":
                faithful += 1
        elif quote and quote in item.passage:
            faithful += 1
    return {
        "schema_validity": schema_ok,
        "quote_faithfulness": round(faithful / len(items), 4),
        "n_cases": float(len(items)),
    }


def main() -> int:
    items = load_items()
    current = score(items)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_metrics = baseline["metrics"]

    regressions: list[str] = []
    for metric, base_value in baseline_metrics.items():
        got = current.get(metric)
        if got is None:
            regressions.append(f"missing metric {metric}")
        elif got < base_value:
            regressions.append(f"{metric}: {got} < baseline {base_value}")

    print(json.dumps({"current": current, "baseline": baseline_metrics}, indent=2))
    if regressions:
        print("REGRESSION:")
        for item in regressions:
            print(f"  {item}")
        return 1

    gate_config = GateConfig.model_validate(baseline.get("gate", {}))
    outcome = run_gate(ROOT / gate_config.results_dir, gate_config)
    print(
        json.dumps(
            {
                "results_gate": outcome.status,
                "results_file": outcome.results_file,
                "checks": outcome.checks,
                "failures": outcome.failures,
            },
            indent=2,
        )
    )
    if outcome.status == "fail":
        print("RESULTS GATE FAILED")
        return 1

    print(
        "Eval gate passed."
        if outcome.status == "pass"
        else "Eval gate passed (results gate skipped)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
