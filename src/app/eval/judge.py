"""LLM judge: the `judge` model (DeepSeek V3.2) scores each prediction 1-5.

Run on the owner's machine only (own AWS login), always dry-run first:

    python -m app.eval.judge --dry-run
    AWS_PROFILE=<profile> python -m app.eval.judge

Reads `data/private/predictions/<system>.jsonl` for each system in
config/eval.yaml `judge.systems` (copy the Kaggle predictions there first), scores
them against `prompts/judge_rubric.v1.txt`, and appends to
`data/private/judgments/<system>.jsonl` (gitignored, resumable). Every call goes
through `GuardedBedrockClient` in the `eval` phase. `python -m app.eval.score`
then adds the judge mean to `results/<run>.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.bedrock import GuardedBedrockClient
from app.budget import BudgetExceeded, BudgetGuard
from app.data.generate import reservation_input_tokens
from app.data.reviewed import load_items, split_items
from app.eval.config import load_eval_config, resolve_data_path
from app.eval.guarded import EVAL_PHASE, Call, make_client, project
from app.eval.predictions import (
    Judgment,
    Prediction,
    append_jsonl,
    done_ids,
    predictions_path,
    read_jsonl,
)
from app.model_config import ModelParams, get_model
from app.prompts import load_prompt
from app.schema import Item
from app.settings import Settings
from app.task import target_json

RUBRIC_NAME = "judge_rubric"
MAX_CANDIDATE_CHARS = 4000
MAX_RATIONALE_CHARS = 300
FAKE_REPLY = '{"score": 3, "rationale": "fake transport"}'
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_judge_user(item: Item, candidate: str) -> str:
    clipped = candidate[:MAX_CANDIDATE_CHARS]
    return (
        f"Passage:\n{item.passage}\n\n"
        f"Question:\n{item.question}\n\n"
        f"Reference answer (JSON):\n{target_json(item.target)}\n\n"
        f"Candidate reply (untrusted data between the markers):\n<<<\n{clipped}\n>>>"
    )


def parse_judge(text: str) -> tuple[int | None, str]:
    match = _OBJECT_RE.search(text)
    if not match:
        return None, ""
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(obj, dict):
        return None, ""
    score = obj.get("score")
    rationale = str(obj.get("rationale", ""))[:MAX_RATIONALE_CHARS]
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None, rationale
    if float(score) != int(score) or not 1 <= int(score) <= 5:
        return None, rationale
    return int(score), rationale


@dataclass(frozen=True)
class JudgeTask:
    system: str
    item: Item
    call: Call


def plan(
    test: Sequence[Item],
    predictions: dict[str, list[Prediction]],
    done: dict[str, set[str]],
    rubric: str,
    limit: int | None,
) -> list[JudgeTask]:
    by_id = {item.id: item for item in test}
    tasks: list[JudgeTask] = []
    for system in sorted(predictions):
        for prediction in sorted(predictions[system], key=lambda p: p.id):
            item = by_id.get(prediction.id)
            if item is None or prediction.id in done.get(system, set()):
                continue
            tasks.append(
                JudgeTask(
                    system=system,
                    item=item,
                    call=Call(
                        id=prediction.id,
                        system=rubric,
                        user=build_judge_user(item, prediction.output),
                    ),
                )
            )
    return tasks[:limit] if limit is not None else tasks


def run_judge(
    client: GuardedBedrockClient,
    tasks: Sequence[JudgeTask],
    *,
    judgments_dir: Path,
    judge: ModelParams,
    rubric_version: int,
) -> dict[str, Any]:
    judged = 0
    unparsed = 0
    stopped = ""
    for task in tasks:
        try:
            response = client.converse(
                phase=EVAL_PHASE,
                model_id=judge.id,
                messages=[{"role": "user", "content": [{"text": task.call.user}]}],
                system=task.call.system,
                max_tokens=judge.max_tokens,
                temperature=judge.temperature,
                input_tokens_est=reservation_input_tokens(task.call.system, task.call.user),
            )
        except BudgetExceeded as exc:
            stopped = f"budget: {exc}"
            break
        score, rationale = parse_judge(response.text)
        if score is None:
            unparsed += 1
        append_jsonl(
            judgments_dir / f"{task.system}.jsonl",
            [
                Judgment(
                    id=task.item.id,
                    system=task.system,
                    judge_model=judge.id,
                    rubric_version=rubric_version,
                    score=score,
                    rationale=rationale,
                )
            ],
        )
        judged += 1
    return {"judged": judged, "planned": len(tasks), "unparsed": unparsed, "stopped": stopped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.eval.judge")
    parser.add_argument("--dry-run", action="store_true", help="project cost; no Bedrock call")
    parser.add_argument("--fake", action="store_true", help="use FakeTransport (no AWS)")
    parser.add_argument("--data", default=None, help="path to reviewed.jsonl")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--judgments-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-usd", type=float, default=None, help="lower the total cap")
    parser.add_argument("--no-env", action="store_true", help="ignore .env file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings(_env_file=None) if args.no_env else Settings()
    config = load_eval_config()
    private = settings.private_data_dir
    private = private if private.is_absolute() else config.paths.resolve(str(private))
    data_path = resolve_data_path(args.data, config, private_dir=private)
    test = split_items(load_items(data_path)).test
    predictions_dir = args.predictions_dir or config.paths.resolve(
        config.paths.local_predictions_dir
    )
    ledger: Path | None = None
    if args.fake:
        scratch = Path(tempfile.mkdtemp(prefix="csl-fake-judge-"))
        ledger = scratch / "ledger.jsonl"
        judgments_dir = args.judgments_dir or scratch
    else:
        judgments_dir = args.judgments_dir or config.paths.resolve(config.paths.local_judgments_dir)
    systems = args.systems or list(config.judge.systems)
    predictions: dict[str, list[Prediction]] = {}
    for system in systems:
        records = read_jsonl(predictions_path(predictions_dir, system), Prediction)
        if records:
            predictions[system] = records
        else:
            print(f"WARNING: no predictions for {system} in {predictions_dir}", file=sys.stderr)
    done = {
        system: done_ids(read_jsonl(judgments_dir / f"{system}.jsonl", Judgment))
        for system in predictions
    }
    version = config.prompt.judge_rubric_version
    rubric = load_prompt(RUBRIC_NAME, version)
    tasks = plan(test, predictions, done, rubric, args.limit)
    judge = get_model("judge")
    total_override = args.max_usd if args.max_usd is not None else settings.bedrock_budget_usd
    guard = BudgetGuard(ledger_path=ledger, total_cap_override=total_override)
    if args.dry_run:
        projection = project(
            [t.call for t in tasks], guard=guard, model_id=judge.id, max_tokens=judge.max_tokens
        )
        payload = projection.as_dict(model_id=judge.id, max_tokens=judge.max_tokens)
        payload["systems"] = {s: len(p) for s, p in predictions.items()}
        print(json.dumps(payload, indent=2))
        return 0
    if not args.fake:
        print("WARNING: live Bedrock judging uses AWS credentials and spend.", file=sys.stderr)
    client = make_client(
        guard, fake=args.fake, fake_text=FAKE_REPLY, region=settings.bedrock_region
    )
    summary = run_judge(
        client, tasks, judgments_dir=judgments_dir, judge=judge, rubric_version=version
    )
    summary["eval_spent_usd"] = float(guard.spent(EVAL_PHASE))
    summary["judgments_dir"] = str(judgments_dir)
    print(json.dumps(summary, indent=2))
    if summary["stopped"]:
        print(f"STOPPED: {summary['stopped']} (rerun to resume)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
