from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from _synthetic import make_items, write_items
from app.bedrock import FakeTransport, GuardedBedrockClient
from app.budget import BudgetGuard
from app.data.reviewed import split_items
from app.eval import bedrock_baseline, judge
from app.eval.config import load_eval_config
from app.eval.guarded import EVAL_PHASE, Call, project
from app.eval.predictions import Judgment, Prediction, read_jsonl
from app.model_config import get_model
from app.task import build_system_prompt

ANSWER = '{"answerable": false, "answer": "", "quote": "", "concepts": []}'


def _client(tmp_path: Path, text: str, cap: float | None = None) -> GuardedBedrockClient:
    guard = BudgetGuard(ledger_path=tmp_path / "ledger.jsonl", total_cap_override=cap)
    return GuardedBedrockClient(guard, FakeTransport(text=text, input_tokens=50, output_tokens=20))


def test_baseline_uses_eval_phase_and_resumes(tmp_path: Path) -> None:
    config = load_eval_config()
    test = split_items(make_items()).test
    system = build_system_prompt()
    planned = bedrock_baseline.plan(test, set(), system, None)
    client = _client(tmp_path, ANSWER)
    out = tmp_path / "preds" / "qwen3-235b.jsonl"
    model_id = get_model("generator").id
    summary = bedrock_baseline.run_baseline(
        client, planned[:2], out_path=out, config=config, model_id=model_id
    )
    assert summary["answered"] == 2 and not summary["stopped"]
    ledger = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert {e["phase"] for e in ledger} == {EVAL_PHASE}
    assert client.guard.spent(EVAL_PHASE) > 0
    transport = client.transport
    assert isinstance(transport, FakeTransport)
    assert transport.calls[0]["max_tokens"] == config.bedrock.answer_max_tokens
    assert transport.calls[0]["system"] == system
    records = read_jsonl(out, Prediction)
    assert [r.system for r in records] == ["qwen3-235b", "qwen3-235b"]
    done = {r.id for r in records}
    assert len(bedrock_baseline.plan(test, done, system, None)) == len(test) - 2


def test_baseline_stops_cleanly_at_cap(tmp_path: Path) -> None:
    config = load_eval_config()
    test = split_items(make_items()).test
    planned = bedrock_baseline.plan(test, set(), build_system_prompt(), None)
    client = _client(tmp_path, ANSWER, cap=0.00000001)
    out = tmp_path / "qwen3-235b.jsonl"
    summary = bedrock_baseline.run_baseline(
        client, planned, out_path=out, config=config, model_id=get_model("generator").id
    )
    assert summary["answered"] == 0
    assert summary["stopped"].startswith("budget")
    assert not out.exists()


def test_projection_fits_eval_cap(tmp_path: Path) -> None:
    guard = BudgetGuard(ledger_path=tmp_path / "ledger.jsonl")
    calls = [Call(id=str(i), system="s " * 200, user="u " * 600) for i in range(107)]
    proj = project(calls, guard=guard, model_id=get_model("generator").id, max_tokens=400)
    assert proj.calls == 107
    assert Decimal("0") < proj.worst_case_usd < proj.eval_cap_usd
    payload = proj.as_dict(model_id="m", max_tokens=400)
    assert payload["fits_remaining_eval_budget"] is True


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ('{"score": 4, "rationale": "fine"}', 4),
        ('Here: {"score": 5, "rationale": "ok"}', 5),
        ('{"score": 4.0, "rationale": "x"}', 4),
        ('{"score": 7, "rationale": "x"}', None),
        ('{"score": true}', None),
        ('{"score": 2.5}', None),
        ("no json", None),
    ],
)
def test_parse_judge(text: str, score: int | None) -> None:
    assert judge.parse_judge(text)[0] == score


def test_judge_run_writes_private_judgments(tmp_path: Path) -> None:
    test = split_items(make_items()).test
    preds = {
        "base": [
            Prediction(id=i.id, system="base", output=ANSWER, model="m", hardware="T4")
            for i in test
        ]
    }
    tasks = judge.plan(test, preds, {"base": {test[0].id}}, "RUBRIC", None)
    assert len(tasks) == len(test) - 1
    assert "untrusted" in tasks[0].call.user
    client = _client(tmp_path, '{"score": 5, "rationale": "abstained correctly"}')
    summary = judge.run_judge(
        client,
        tasks,
        judgments_dir=tmp_path / "judgments",
        judge=get_model("judge"),
        rubric_version=1,
    )
    assert summary["judged"] == len(tasks) and summary["unparsed"] == 0
    records = read_jsonl(tmp_path / "judgments" / "base.jsonl", Judgment)
    assert {r.score for r in records} == {5}
    transport = client.transport
    assert isinstance(transport, FakeTransport)
    assert transport.calls[0]["model_id"] == get_model("judge").id
    assert transport.calls[0]["system"] == "RUBRIC"


def test_cli_dry_run_makes_no_calls(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = write_items(tmp_path / "reviewed.jsonl", make_items())
    code = bedrock_baseline.main(
        ["--dry-run", "--no-env", "--data", str(data), "--out-dir", str(tmp_path / "p")]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True and payload["calls"] == 5
    assert not (tmp_path / "p").exists()


def test_cli_fake_uses_scratch_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from app.paths import repo_root

    ledger = repo_root() / "data" / "ledger" / "bedrock_spend.jsonl"
    before = ledger.read_text(encoding="utf-8")
    data = write_items(tmp_path / "reviewed.jsonl", make_items())
    assert bedrock_baseline.main(["--fake", "--no-env", "--data", str(data), "--limit", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answered"] == 2
    assert ledger.read_text(encoding="utf-8") == before


def test_judge_rubric_prompt_loads() -> None:
    from app.prompts import load_prompt

    text = load_prompt("judge_rubric", 1)
    assert "untrusted" in text.lower()
    assert '"score"' in text


def test_judge_cli_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = make_items()
    data = write_items(tmp_path / "reviewed.jsonl", items)
    preds_dir = tmp_path / "preds"
    from app.eval.predictions import append_jsonl

    test = split_items(items).test
    append_jsonl(
        preds_dir / "base.jsonl",
        [Prediction(id=i.id, system="base", output=ANSWER, model="m", hardware="T4") for i in test],
    )
    code = judge.main(
        [
            "--dry-run",
            "--no-env",
            "--data",
            str(data),
            "--predictions-dir",
            str(preds_dir),
            "--judgments-dir",
            str(tmp_path / "j"),
            "--systems",
            "base",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["calls"] == len(test)
    assert payload["model_id"] == get_model("judge").id
    assert not (tmp_path / "j").exists()
