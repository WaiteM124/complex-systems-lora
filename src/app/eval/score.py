"""Score prediction files into `results/<run>.json` (aggregates only).

Run: `python -m app.eval.score --run <name> [--predictions-dir D] [--results-dir R]`

Every `<system>.jsonl` in the predictions directory is scored against the test
split; judgments in `--judgments-dir` (if any) add the judge mean. Re-running
updates the same run file, so Kaggle (base, lora) and local Bedrock (qwen3-235b,
judge) results merge into one run as long as the test split is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.data.reviewed import file_sha256, items_sha256, load_items, split_items
from app.eval.config import EvalConfig, load_eval_config, resolve_data_path
from app.eval.predictions import Judgment, Prediction, read_jsonl
from app.eval.results import (
    RunResults,
    SystemResult,
    load_results,
    merge_system,
    score_system,
    summarise_judgments,
    write_results,
)
from app.paths import repo_root
from app.schema import Item


def score_run(
    *,
    run: str,
    data_path: Path,
    predictions_dir: Path,
    results_dir: Path,
    judgments_dir: Path | None,
    config: EvalConfig,
    systems: list[str] | None = None,
    allow_partial: bool = False,
) -> RunResults:
    items = load_items(data_path)
    test: list[Item] = split_items(items).test
    if not test:
        raise ValueError("dataset has no test-split items")
    test_ids = {item.id for item in test}
    dataset_sha = file_sha256(data_path)
    test_sha = items_sha256(test)
    out_path = results_dir / f"{run}.json"
    results = load_results(out_path) if out_path.is_file() else None
    files = sorted(predictions_dir.glob("*.jsonl")) if predictions_dir.is_dir() else []
    if systems:
        files = [f for f in files if f.stem in systems]
    if not files:
        raise FileNotFoundError(f"no prediction files in {predictions_dir}")
    for path in files:
        predictions = read_jsonl(path, Prediction)
        if not predictions:
            continue
        system = path.stem
        if any(p.system != system for p in predictions):
            raise ValueError(f"{path.name}: records name a different system")
        metrics = score_system(test, predictions, allow_partial=allow_partial)
        judge = None
        if judgments_dir is not None:
            judge = summarise_judgments(
                read_jsonl(judgments_dir / f"{system}.jsonl", Judgment), test_ids
            )
        first = predictions[0]
        result = SystemResult(
            model=first.model,
            adapter=first.adapter,
            hardware=first.hardware,
            metrics=metrics,
            judge=judge,
        )
        results = merge_system(
            results,
            run=run,
            dataset_sha256=dataset_sha,
            test_split_sha256=test_sha,
            n_test_items=len(test),
            answer_prompt_version=config.prompt.answer_system_version,
            system=system,
            result=result,
        )
    if results is None:
        raise FileNotFoundError(f"no non-empty prediction files in {predictions_dir}")
    write_results(out_path, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.eval.score")
    parser.add_argument("--run", required=True, help="run name, e.g. m3-qlora-r16")
    parser.add_argument("--data", default=None, help="path to reviewed.jsonl")
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--judgments-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--kaggle", action="store_true", help="use Kaggle default paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_eval_config()
    paths = config.paths
    if args.kaggle:
        predictions_dir = args.predictions_dir or Path(paths.kaggle_predictions_dir)
        results_dir = args.results_dir or Path(paths.kaggle_results_dir)
        judgments_dir = args.judgments_dir
    else:
        predictions_dir = args.predictions_dir or paths.resolve(paths.local_predictions_dir)
        results_dir = args.results_dir or paths.resolve(paths.results_dir)
        judgments_dir = args.judgments_dir or paths.resolve(paths.local_judgments_dir)
    data_path = resolve_data_path(args.data, config, private_dir=repo_root() / "data" / "private")
    results = score_run(
        run=args.run,
        data_path=data_path,
        predictions_dir=predictions_dir,
        results_dir=results_dir,
        judgments_dir=judgments_dir,
        config=config,
        systems=args.systems,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(results.model_dump(mode="json"), indent=2, sort_keys=True))
    print(f"wrote {results_dir / (args.run + '.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
