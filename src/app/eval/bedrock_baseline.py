"""System (c): the `generator` model (Qwen3 235B) on Bedrock answers the test split.

Run on the owner's machine only (own AWS login), always dry-run first:

    python -m app.eval.bedrock_baseline --dry-run
    AWS_PROFILE=<profile> python -m app.eval.bedrock_baseline

Every call goes through `GuardedBedrockClient` in the `eval` phase ($3.00 cap in
config/budget.yaml). Predictions append per item to
`data/private/predictions/qwen3-235b.jsonl` (gitignored), so a stop at the cap or
Ctrl-C is resumable. Score with `python -m app.eval.score --run <name>`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.bedrock import GuardedBedrockClient
from app.budget import BudgetExceeded, BudgetGuard
from app.data.generate import reservation_input_tokens
from app.data.reviewed import load_items, split_items
from app.eval.config import EvalConfig, load_eval_config, resolve_data_path
from app.eval.guarded import EVAL_PHASE, Call, make_client, project
from app.eval.predictions import Prediction, append_jsonl, done_ids, predictions_path, read_jsonl
from app.model_config import get_model
from app.schema import Item
from app.settings import Settings
from app.task import build_system_prompt, build_user_prompt

FAKE_REPLY = '{"answerable": false, "answer": "", "quote": "", "concepts": []}'


def plan(
    test: Sequence[Item], done: set[str], system_prompt: str, limit: int | None
) -> list[tuple[Item, Call]]:
    todo = [item for item in sorted(test, key=lambda i: i.id) if item.id not in done]
    if limit is not None:
        todo = todo[:limit]
    return [
        (
            item,
            Call(
                id=item.id,
                system=system_prompt,
                user=build_user_prompt(item.passage, item.question),
            ),
        )
        for item in todo
    ]


def run_baseline(
    client: GuardedBedrockClient,
    planned: Sequence[tuple[Item, Call]],
    *,
    out_path: Path,
    config: EvalConfig,
    model_id: str,
) -> dict[str, Any]:
    done = 0
    stopped = ""
    for item, call in planned:
        started = time.perf_counter()
        try:
            response = client.converse(
                phase=EVAL_PHASE,
                model_id=model_id,
                messages=[{"role": "user", "content": [{"text": call.user}]}],
                system=call.system,
                max_tokens=config.bedrock.answer_max_tokens,
                temperature=config.bedrock.temperature,
                input_tokens_est=reservation_input_tokens(call.system, call.user),
            )
        except BudgetExceeded as exc:
            stopped = f"budget: {exc}"
            break
        latency_ms = (time.perf_counter() - started) * 1000.0
        append_jsonl(
            out_path,
            [
                Prediction(
                    id=item.id,
                    system=config.bedrock.system_name,
                    output=response.text,
                    model=model_id,
                    hardware=config.bedrock.hardware_label,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            ],
        )
        done += 1
    return {"answered": done, "planned": len(planned), "stopped": stopped, "path": str(out_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.eval.bedrock_baseline")
    parser.add_argument("--dry-run", action="store_true", help="project cost; no Bedrock call")
    parser.add_argument("--fake", action="store_true", help="use FakeTransport (no AWS)")
    parser.add_argument("--data", default=None, help="path to reviewed.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None)
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
    ledger: Path | None = None
    if args.fake:
        # Fake runs never touch the committed ledger or the private predictions.
        scratch = Path(tempfile.mkdtemp(prefix="csl-fake-eval-"))
        ledger = scratch / "ledger.jsonl"
        out_dir = args.out_dir or scratch
    else:
        out_dir = args.out_dir or config.paths.resolve(config.paths.local_predictions_dir)
    out_path = predictions_path(out_dir, config.bedrock.system_name)
    test = split_items(load_items(data_path)).test
    system_prompt = build_system_prompt(version=config.prompt.answer_system_version)
    planned = plan(test, done_ids(read_jsonl(out_path, Prediction)), system_prompt, args.limit)
    model_id = get_model("generator").id
    total_override = args.max_usd if args.max_usd is not None else settings.bedrock_budget_usd
    guard = BudgetGuard(ledger_path=ledger, total_cap_override=total_override)
    if args.dry_run:
        projection = project(
            [call for _, call in planned],
            guard=guard,
            model_id=model_id,
            max_tokens=config.bedrock.answer_max_tokens,
        )
        payload = projection.as_dict(model_id=model_id, max_tokens=config.bedrock.answer_max_tokens)
        payload["test_items"] = len(test)
        print(json.dumps(payload, indent=2))
        return 0
    if not args.fake:
        print("WARNING: live Bedrock eval uses AWS credentials and spend.", file=sys.stderr)
    client = make_client(
        guard, fake=args.fake, fake_text=FAKE_REPLY, region=settings.bedrock_region
    )
    summary = run_baseline(client, planned, out_path=out_path, config=config, model_id=model_id)
    summary["eval_spent_usd"] = float(guard.spent(EVAL_PHASE))
    print(json.dumps(summary, indent=2))
    if summary["stopped"]:
        print(f"STOPPED: {summary['stopped']} (rerun to resume)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
