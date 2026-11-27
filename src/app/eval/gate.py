"""Results gate for `make eval`: base vs base + LoRA on the hand-checked test split.

Starting bars from the build spec (revise after the baseline, never silently):

- Primary: LoRA answerability F1 >= base + 10 points. If base F1 is already above
  85, the primary switches to quote faithfulness >= base + 10 points.
- Must not get worse: quote faithfulness >= 90%, JSON validity >= 98%,
  judge mean no more than 0.2 below base (checked once both are judged).

With no results committed, or only a baseline, the gate skips cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.eval.results import RunResults, SystemResult, load_results

EPS = 1e-9


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: str = "results"
    base_system: str = "base"
    lora_system: str = "lora"
    primary_metric: str = "answerability_f1"
    primary_min_gain: float = 0.10
    base_primary_switch_above: float = 0.85
    switched_primary_metric: str = "quote_faithfulness"
    quote_faithfulness_min: float = 0.90
    json_validity_min: float = 0.98
    judge_max_drop: float = 0.2


@dataclass
class GateOutcome:
    status: Literal["pass", "fail", "skip"]
    results_file: str | None = None
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _metric(system: SystemResult, name: str) -> float | None:
    value = system.metrics.get(name)
    return None if value is None else float(value)


def evaluate_run(results: RunResults, config: GateConfig) -> GateOutcome:
    outcome = GateOutcome(status="pass", results_file=results.run)
    base = results.systems.get(config.base_system)
    lora = results.systems.get(config.lora_system)
    if base is None or lora is None:
        missing = config.base_system if base is None else config.lora_system
        outcome.status = "skip"
        outcome.checks.append(f"system {missing!r} not in run {results.run!r}; gate skipped")
        return outcome

    def need(ok: bool, message: str) -> None:
        (outcome.checks if ok else outcome.failures).append(message)

    for name, system in ((config.base_system, base), (config.lora_system, lora)):
        coverage = _metric(system, "coverage")
        need(
            coverage is not None and coverage >= 1.0 - EPS,
            f"{name} coverage of the test split = {coverage}",
        )

    base_f1 = _metric(base, config.primary_metric)
    primary = config.primary_metric
    if base_f1 is not None and base_f1 > config.base_primary_switch_above:
        primary = config.switched_primary_metric
        outcome.checks.append(
            f"base {config.primary_metric} {base_f1:.4f} > "
            f"{config.base_primary_switch_above}: primary switches to {primary}"
        )
    b, lo = _metric(base, primary), _metric(lora, primary)
    if b is None or lo is None:
        need(False, f"primary {primary}: missing value (base={b}, lora={lo})")
    else:
        need(
            lo + EPS >= b + config.primary_min_gain,
            f"primary {primary}: lora {lo:.4f} vs base {b:.4f} "
            f"(needs +{config.primary_min_gain:.2f})",
        )

    qf = _metric(lora, "quote_faithfulness")
    need(
        qf is not None and qf + EPS >= config.quote_faithfulness_min,
        f"lora quote_faithfulness {qf} >= {config.quote_faithfulness_min}",
    )
    jv = _metric(lora, "json_validity")
    need(
        jv is not None and jv + EPS >= config.json_validity_min,
        f"lora json_validity {jv} >= {config.json_validity_min}",
    )

    base_judge = base.judge.mean if base.judge else None
    lora_judge = lora.judge.mean if lora.judge else None
    if base_judge is None or lora_judge is None:
        outcome.checks.append("judge scores not recorded for both systems yet; judge check skipped")
    else:
        need(
            lora_judge + EPS >= base_judge - config.judge_max_drop,
            f"lora judge {lora_judge:.3f} vs base {base_judge:.3f} "
            f"(max drop {config.judge_max_drop})",
        )

    if outcome.failures:
        outcome.status = "fail"
    return outcome


def run_gate(results_dir: Path, config: GateConfig) -> GateOutcome:
    """Validate every results file, then gate the newest run holding base and LoRA."""
    files = sorted(results_dir.glob("*.json")) if results_dir.is_dir() else []
    if not files:
        return GateOutcome(status="skip", checks=[f"no results in {results_dir.name}/ yet"])
    loaded: list[tuple[str, str, RunResults]] = []
    invalid: list[str] = []
    for path in files:
        try:
            run = load_results(path)
        except (ValidationError, ValueError) as exc:
            invalid.append(f"{path.name}: invalid results file ({type(exc).__name__})")
            continue
        loaded.append((run.updated_at, path.name, run))
    if invalid:
        return GateOutcome(status="fail", failures=invalid)
    complete = [
        entry
        for entry in loaded
        if config.base_system in entry[2].systems and config.lora_system in entry[2].systems
    ]
    if not complete:
        newest = sorted(loaded, key=lambda e: (e[0], e[1]))[-1]
        outcome = evaluate_run(newest[2], config)
        outcome.results_file = newest[1]
        return outcome
    _, name, run = sorted(complete, key=lambda e: (e[0], e[1]))[-1]
    outcome = evaluate_run(run, config)
    outcome.results_file = name
    return outcome
