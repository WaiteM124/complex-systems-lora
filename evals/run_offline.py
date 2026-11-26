#!/usr/bin/env python3
"""Deterministic offline eval gate.

Reads every JSONL file in evals/datasets/, scores cases against
evals/baseline.json, and exits non-zero on regression. The example
dataset is solved with a trivial local predictor so CI is green from
day one. Replace `predict` with the project's real, still-deterministic
logic when you add model or metric work. Live or keyed evals belong in
a manually dispatched workflow, not this gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "evals" / "baseline.json"
DATASETS_DIR = ROOT / "evals" / "datasets"


class Case(TypedDict):
    id: str
    input: str
    expected: str
    tags: list[str]
    source: str
    licence: str
    predicted: str


def load_cases(directory: Path = DATASETS_DIR) -> list[Case]:
    cases: list[Case] = []
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL datasets in {directory}")
    for path in paths:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = {"id", "input", "expected", "tags", "source", "licence"} - set(item)
            if missing:
                raise ValueError(f"{path.name}:{line_no} missing fields {sorted(missing)}")
            cases.append(
                {
                    "id": item["id"],
                    "input": item["input"],
                    "expected": item["expected"],
                    "tags": list(item["tags"]),
                    "source": item["source"],
                    "licence": item["licence"],
                    "predicted": "",
                }
            )
    return cases


def predict(case: Case) -> str:
    """Trivial local predictor. Replace with deterministic project logic."""
    if "uppercase" in case["tags"]:
        return case["input"].upper()
    return case["input"]


def score(cases: list[Case]) -> dict[str, float]:
    if not cases:
        raise ValueError("no eval cases")
    exact = sum(1 for case in cases if case["expected"] == case["predicted"]) / len(cases)
    return {"exact_match": round(exact, 4), "n_cases": float(len(cases))}


def main() -> int:
    cases = load_cases()
    for case in cases:
        case["predicted"] = predict(case)
    current = score(cases)
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

    print("Eval gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
