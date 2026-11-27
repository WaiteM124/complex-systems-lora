from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from _synthetic import answer_text, make_items, write_items
from app.data.reviewed import split_items
from app.eval.config import load_eval_config, resolve_data_path
from app.eval.gate import GateConfig, evaluate_run, run_gate
from app.eval.predictions import Prediction, append_jsonl
from app.eval.results import (
    CoverageError,
    JudgeSummary,
    RunResults,
    SystemResult,
    merge_system,
    score_system,
    write_results,
)
from app.eval.score import score_run
from app.paths import repo_root

SHA = "a" * 64
ABSTAIN = '{"answerable": false, "answer": "", "quote": "", "concepts": []}'


def _pred(item_id: str, system: str, output: str) -> Prediction:
    return Prediction(
        id=item_id,
        system=system,
        output=output,
        model="base/model",
        hardware="Tesla T4",
        latency_ms=100.0,
        output_tokens=20,
    )


def _system(**metrics: float | None) -> SystemResult:
    base = {"coverage": 1.0, "json_validity": 1.0, "quote_faithfulness": 0.95}
    base.update(metrics)  # type: ignore[arg-type]
    return SystemResult(model="m", hardware="Tesla T4", metrics=base)


def _run(base: SystemResult | None, lora: SystemResult | None, stamp: str = "t1") -> RunResults:
    systems = {}
    if base:
        systems["base"] = base
    if lora:
        systems["lora"] = lora
    return RunResults(
        run="r1",
        created_at=stamp,
        updated_at=stamp,
        dataset_sha256=SHA,
        test_split_sha256=SHA,
        n_test_items=5,
        answer_prompt_version=1,
        systems=systems,
    )


def test_score_system_coverage() -> None:
    test = split_items(make_items()).test
    perfect = [_pred(i.id, "base", answer_text(i)) for i in test]
    metrics = score_system(test, perfect)
    assert metrics["answerability_f1"] == 1.0
    assert metrics["quote_faithfulness"] == 1.0
    assert metrics["coverage"] == 1.0
    with pytest.raises(CoverageError):
        score_system(test, perfect[:-1])
    assert score_system(test, perfect[:-1], allow_partial=True)["coverage"] == 0.8
    with pytest.raises(CoverageError):
        score_system(test, [*perfect, _pred("not-a-test-id", "base", ABSTAIN)])


def test_results_forbid_text_fields() -> None:
    with pytest.raises(ValidationError):
        SystemResult(model="m", hardware="h", metrics={}, output="leak")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        _run(None, None).model_validate({**_run(None, None).model_dump(), "passages": ["x"]})
    with pytest.raises(ValidationError):
        RunResults.model_validate({**_run(None, None).model_dump(), "test_split_sha256": "nope"})


def test_merge_refuses_other_test_split() -> None:
    run = _run(_system(answerability_f1=0.5), None)
    merged = merge_system(
        run,
        run="r1",
        dataset_sha256=SHA,
        test_split_sha256=SHA,
        n_test_items=5,
        answer_prompt_version=1,
        system="lora",
        result=_system(answerability_f1=0.7),
    )
    assert set(merged.systems) == {"base", "lora"}
    with pytest.raises(ValueError, match="different test split"):
        merge_system(
            run,
            run="r1",
            dataset_sha256=SHA,
            test_split_sha256="b" * 64,
            n_test_items=5,
            answer_prompt_version=1,
            system="lora",
            result=_system(),
        )


def test_gate_pass_and_primary_fail() -> None:
    cfg = GateConfig()
    ok = evaluate_run(_run(_system(answerability_f1=0.6), _system(answerability_f1=0.7)), cfg)
    assert ok.status == "pass", ok.failures
    short = evaluate_run(_run(_system(answerability_f1=0.6), _system(answerability_f1=0.69)), cfg)
    assert short.status == "fail"
    assert any("primary answerability_f1" in f for f in short.failures)


def test_gate_switches_primary_when_base_is_strong() -> None:
    cfg = GateConfig()
    base = _system(answerability_f1=0.9, quote_faithfulness=0.80)
    lora = _system(answerability_f1=0.9, quote_faithfulness=0.95)
    outcome = evaluate_run(_run(base, lora), cfg)
    assert outcome.status == "pass", outcome.failures
    assert any("switches to quote_faithfulness" in c for c in outcome.checks)
    weak = evaluate_run(_run(base, _system(answerability_f1=0.99, quote_faithfulness=0.85)), cfg)
    assert weak.status == "fail"


def test_gate_must_not_get_worse_checks() -> None:
    cfg = GateConfig()
    base = _system(answerability_f1=0.5)
    bad_json = evaluate_run(_run(base, _system(answerability_f1=0.8, json_validity=0.97)), cfg)
    assert bad_json.status == "fail"
    bad_quotes = evaluate_run(
        _run(base, _system(answerability_f1=0.8, quote_faithfulness=0.89)), cfg
    )
    assert bad_quotes.status == "fail"
    partial = evaluate_run(_run(base, _system(answerability_f1=0.8, coverage=0.9)), cfg)
    assert partial.status == "fail"


def test_gate_judge_check() -> None:
    cfg = GateConfig()
    base = _system(answerability_f1=0.5)
    lora = _system(answerability_f1=0.8)
    skipped = evaluate_run(_run(base, lora), cfg)
    assert skipped.status == "pass"
    assert any("judge check skipped" in c for c in skipped.checks)

    def judged(system: SystemResult, mean: float) -> SystemResult:
        summary = JudgeSummary(model="j", rubric_version=1, mean=mean, n=5, n_unparsed=0)
        return system.model_copy(update={"judge": summary})

    ok = evaluate_run(_run(judged(base, 4.0), judged(lora, 3.85)), cfg)
    assert ok.status == "pass", ok.failures
    drop = evaluate_run(_run(judged(base, 4.0), judged(lora, 3.7)), cfg)
    assert drop.status == "fail"


def test_run_gate_skips_and_validates(tmp_path: Path) -> None:
    cfg = GateConfig()
    assert run_gate(tmp_path / "none", cfg).status == "skip"
    write_results(tmp_path / "baseline.json", _run(_system(answerability_f1=0.5), None))
    assert run_gate(tmp_path, cfg).status == "skip"
    write_results(
        tmp_path / "tuned.json",
        _run(_system(answerability_f1=0.5), _system(answerability_f1=0.7), stamp="t2"),
    )
    outcome = run_gate(tmp_path, cfg)
    assert (outcome.status, outcome.results_file) == ("pass", "tuned.json")
    (tmp_path / "broken.json").write_text('{"run": "x"}', encoding="utf-8")
    assert run_gate(tmp_path, cfg).status == "fail"


def test_score_run_end_to_end(tmp_path: Path) -> None:
    items = make_items()
    data = write_items(tmp_path / "private" / "reviewed.jsonl", items)
    test = split_items(items).test
    preds = tmp_path / "predictions"
    append_jsonl(preds / "base.jsonl", [_pred(i.id, "base", ABSTAIN) for i in test])
    append_jsonl(preds / "lora.jsonl", [_pred(i.id, "lora", answer_text(i)) for i in test])
    results_dir = tmp_path / "results"
    results = score_run(
        run="test-run",
        data_path=data,
        predictions_dir=preds,
        results_dir=results_dir,
        judgments_dir=None,
        config=load_eval_config(),
    )
    assert set(results.systems) == {"base", "lora"}
    assert results.systems["lora"].metrics["answerability_f1"] == 1.0
    written = (results_dir / "test-run.json").read_text(encoding="utf-8")
    for item in items:
        assert item.passage not in written
        assert item.question not in written
    assert run_gate(results_dir, GateConfig()).status == "pass"


def test_resolve_data_path_prefers_local_private(tmp_path: Path) -> None:
    path = write_items(tmp_path / "reviewed.jsonl", make_items())
    config = load_eval_config()
    assert resolve_data_path(None, config, private_dir=tmp_path) == path
    assert resolve_data_path(str(path), config) == path


def test_offline_gate_passes_in_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(repo_root() / "evals" / "run_offline.py")],
        capture_output=True,
        text=True,
        env={"APP_ENV": "local"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "results_gate" in result.stdout


def test_baseline_gate_thresholds_match_spec() -> None:
    raw = json.loads((repo_root() / "evals" / "baseline.json").read_text(encoding="utf-8"))
    cfg = GateConfig.model_validate(raw["gate"])
    assert cfg.primary_min_gain == pytest.approx(0.10)
    assert cfg.base_primary_switch_above == pytest.approx(0.85)
    assert cfg.quote_faithfulness_min == pytest.approx(0.90)
    assert cfg.json_validity_min == pytest.approx(0.98)
    assert cfg.judge_max_drop == pytest.approx(0.2)
