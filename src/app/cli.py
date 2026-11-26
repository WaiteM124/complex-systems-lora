"""CLI subcommands for the data pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.bedrock import (
    BedrockTransport,
    Boto3ConverseTransport,
    FakeTransport,
    GuardedBedrockClient,
)
from app.budget import BudgetGuard
from app.data.chunk import Chunk, chunk_text, load_data_config
from app.data.export_review import export_review_csv
from app.data.extract import extract_file
from app.data.fetch import fetch_all, format_summary_table, raw_dir
from app.data.filters import filter_items
from app.data.generate import generate_items, project_dry_run
from app.data.import_review import import_review_csv
from app.data.leakage import assert_items_no_leakage
from app.data.licence import write_routed_items
from app.data.notes import load_notes_dir, notes_to_chunks
from app.data.sources import get_source, ingestable_sources, load_sources, require_ingestable
from app.data.split import assign_sections, split_chunks, write_assignment_json, write_test_sections
from app.model_config import get_region
from app.paths import repo_root
from app.schema import Item
from app.settings import Settings


def _private_root(settings: Settings) -> Path:
    path = settings.private_data_dir
    return path if path.is_absolute() else repo_root() / path


def cmd_sources(_: argparse.Namespace) -> int:
    data = load_sources()
    ingest = [s.id for s in data.sources if s.ingest]
    print(json.dumps({"count": len(data.sources), "ingest": ingest}, indent=2))
    return 0


def cmd_fetch(_: argparse.Namespace) -> int:
    results = fetch_all()
    print(format_summary_table(results))
    for row in results:
        if row.status == "failed":
            print(f"WARNING: {row.source_id}: {row.reason}", file=sys.stderr)
    failed = sum(1 for row in results if row.status == "failed")
    ok = sum(1 for row in results if row.status == "ok")
    if ok == 0:
        return 1
    return 1 if failed else 0


def cmd_chunk(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    raw = raw_dir(private)
    chunks_out = private / "chunks.jsonl"
    chunks_out.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []
    sources_chunked = 0
    for source in ingestable_sources():
        require_ingestable(source.id)
        # Pathlib glob is Windows-safe; match source_id.* in the raw dir only.
        matches = sorted(path for path in raw.glob(f"{source.id}.*") if path.is_file())
        if not matches:
            print(f"WARNING: {source.id}: no raw file in {raw}", file=sys.stderr)
            continue
        try:
            text = extract_file(matches[0], source.format)
            produced = chunk_text(text, source_id=source.id)
        except Exception as exc:  # noqa: BLE001 — continue past one bad source
            print(
                f"WARNING: {source.id}: skipping after extract/chunk error: {exc}",
                file=sys.stderr,
            )
            continue
        if not produced:
            print(f"WARNING: {source.id}: produced zero chunks", file=sys.stderr)
            continue
        all_chunks.extend(produced)
        sources_chunked += 1
    notes_dir = private / "notes"
    if notes_dir.exists():
        try:
            note_chunks = notes_to_chunks(load_notes_dir(notes_dir))
            all_chunks.extend(note_chunks)
            if note_chunks:
                sources_chunked += 1
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: notes: skipping after error: {exc}", file=sys.stderr)
    with chunks_out.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in all_chunks:
            handle.write(
                json.dumps(
                    {
                        "section_path": chunk.section_path,
                        "section_id": chunk.section_id,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "chunk_hash": chunk.chunk_hash,
                        "chunk_index": chunk.chunk_index,
                        "source_id": chunk.source_id,
                    }
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "chunks": len(all_chunks),
                "sources_chunked": sources_chunked,
                "path": str(chunks_out),
            }
        )
    )
    if sources_chunked == 0:
        print("ERROR: zero sources were chunked", file=sys.stderr)
        return 1
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    chunks_path = private / "chunks.jsonl"
    from app.data.chunk import Chunk

    chunks: list[Chunk] = []
    if chunks_path.exists():
        for raw in chunks_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            chunks.append(Chunk(**obj))
    assignment = assign_sections([c.section_id for c in chunks])
    write_test_sections(assignment)
    write_assignment_json(assignment)
    split = split_chunks(chunks)
    print(
        json.dumps(
            {
                "train_sections": sum(1 for s in assignment.values() if s == "train"),
                "val_sections": sum(1 for s in assignment.values() if s == "val"),
                "test_sections": sum(1 for s in assignment.values() if s == "test"),
                "train_chunks": len(split["train"]),
                "val_chunks": len(split["val"]),
                "test_chunks": len(split["test"]),
            },
            indent=2,
        )
    )
    return 0


def _load_split_chunks(private: Path) -> list[tuple[Chunk, str, str]]:
    chunks_path = private / "chunks.jsonl"
    assignment_path = repo_root() / "data" / "splits" / "section_assignment.json"
    if assignment_path.exists():
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    else:
        assignment = {}
    out: list[tuple[Chunk, str, str]] = []
    if not chunks_path.exists():
        return out
    for raw in chunks_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        chunk = Chunk(**obj)
        split = assignment.get(chunk.section_id, "train")
        if chunk.source_id == "notes":
            licence = "N"
        else:
            licence = get_source(chunk.source_id).licence_class
        out.append((chunk, split, licence))
    return out


def cmd_generate(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    ledger = repo_root() / "data" / "ledger" / "bedrock_spend.jsonl"
    total_override = args.max_usd if args.max_usd is not None else settings.bedrock_budget_usd
    guard = BudgetGuard(ledger_path=ledger, total_cap_override=total_override)
    entries = _load_split_chunks(private)
    # Prefer fixture chunks when none are available (local dry-run on fixtures).
    if not entries and args.fixtures:
        entries = _fixture_entries()
    config = load_data_config()
    over = float(config.get("generation", {}).get("over_generation", 1.4))
    n = max(1, int(round(len(entries) * over))) if entries else 0
    selected = entries[:n]

    if args.dry_run:
        estimate = project_dry_run([e[0] for e in selected], guard=guard)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "call_count": estimate.call_count,
                    "worst_case_cost_usd": float(estimate.worst_case_cost_usd),
                    "model_id": estimate.model_id,
                    "max_tokens": estimate.max_tokens,
                    "estimated_input_tokens_per_call": estimate.estimated_input_tokens_per_call,
                },
                indent=2,
            )
        )
        return 0

    transport: BedrockTransport
    if args.fake:
        transport = FakeTransport(text='{"items": []}')
    else:
        region = settings.bedrock_region or get_region()
        transport = Boto3ConverseTransport(region=region)
    client = GuardedBedrockClient(guard, transport)
    print(
        "WARNING: live generation uses AWS credentials and Bedrock spend.",
        file=sys.stderr,
    )
    generated = generate_items(selected, client, dry_run=False)
    assert isinstance(generated, list)
    out_path = private / "generated.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in generated:
            handle.write(item.model_dump_json() + "\n")
    print(json.dumps({"items": len(generated), "path": str(out_path)}))
    return 0


def _fixture_entries() -> list[tuple[Chunk, str, str]]:
    """Build dry-run chunks from local fixtures for tier-A ingest sources only."""
    fixture_root = repo_root() / "tests" / "fixtures"
    format_to_fixture = {
        "arxiv_src": fixture_root / "arxiv_src" / "sample.tex",
        "html": fixture_root / "html" / "sample.html",
        "jats": fixture_root / "html" / "sample.html",
        "libretexts": fixture_root / "libretexts" / "sample.html",
        "pdf": fixture_root / "pdf" / "sample.pdf",
    }
    entries: list[tuple[Chunk, str, str]] = []
    for source in ingestable_sources():
        if source.licence_class != "A":
            continue
        path = format_to_fixture.get(source.format)
        if path is None or not path.exists():
            continue
        text = extract_file(path, source.format)
        for chunk in chunk_text(text, source_id=source.id, min_tokens=1, max_tokens=500):
            entries.append((chunk, "train", source.licence_class))
    return entries


def cmd_filter(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    src = private / "generated.jsonl"
    raw_items = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_items.append(json.loads(line))
    filtered = filter_items(raw_items)
    out = private / "filtered.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for item in filtered:
            handle.write(item.model_dump_json() + "\n")
    write_routed_items(filtered, private_root=private)
    print(json.dumps({"input": len(raw_items), "filtered": len(filtered)}))
    return 0


def cmd_leakage(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    path = private / "filtered.jsonl"
    items: list[Item] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(Item.model_validate_json(line))
    assert_items_no_leakage(items)
    print(json.dumps({"ok": True, "items": len(items)}))
    return 0


def cmd_export_review(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    path = private / "filtered.jsonl"
    items: list[Item] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(Item.model_validate_json(line))
    out = private / "review" / "test_review.csv"
    export_review_csv(items, out)
    print(json.dumps({"rows": sum(1 for i in items if i.split == "test"), "path": str(out)}))
    return 0


def cmd_import_review(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    private = _private_root(settings)
    review = private / "review" / "test_review.csv"
    existing_path = private / "filtered.jsonl"
    existing: list[Item] = []
    if existing_path.exists():
        for line in existing_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(Item.model_validate_json(line))
    items = import_review_csv(review, existing=existing)
    out = private / "reviewed.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")
    write_routed_items(items, private_root=private)
    print(json.dumps({"items": len(items), "path": str(out)}))
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    settings = Settings(_env_file=None) if args.no_env else Settings()
    guard = BudgetGuard(total_cap_override=settings.bedrock_budget_usd)
    payload = {
        "total_cap": float(guard.total_cap),
        "phase_caps": {k: float(v) for k, v in guard.phase_caps.items()},
        "spent_total": float(guard.spent()),
        "spent_generation": float(guard.spent("generation")),
        "spent_eval": float(guard.spent("eval")),
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app")
    parser.add_argument("--no-env", action="store_true", help="ignore .env file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="list sources").set_defaults(func=cmd_sources)
    sub.add_parser("fetch", help="download raw sources").set_defaults(func=cmd_fetch)
    p_chunk = sub.add_parser("chunk", help="extract and chunk sources")
    p_chunk.set_defaults(func=cmd_chunk)
    p_split = sub.add_parser("split", help="split sections before generation")
    p_split.set_defaults(func=cmd_split)

    p_gen = sub.add_parser("generate", help="generate items (budget-guarded)")
    p_gen.add_argument("--dry-run", action="store_true")
    p_gen.add_argument("--max-usd", type=float, default=None)
    p_gen.add_argument("--fake", action="store_true", help="use FakeTransport")
    p_gen.add_argument(
        "--fixtures",
        action="store_true",
        help="use tests/fixtures when private chunks are absent",
    )
    p_gen.set_defaults(func=cmd_generate)

    sub.add_parser("filter", help="filter generated items").set_defaults(func=cmd_filter)
    sub.add_parser("leakage", help="check train/test leakage").set_defaults(func=cmd_leakage)
    sub.add_parser("export-review", help="export test CSV").set_defaults(func=cmd_export_review)
    sub.add_parser("import-review", help="import reviewed CSV").set_defaults(func=cmd_import_review)
    sub.add_parser("budget", help="show spend vs caps").set_defaults(func=cmd_budget)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
