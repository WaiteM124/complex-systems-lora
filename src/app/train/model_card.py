"""Model card for the published adapter. Lists every training source and licence."""

from __future__ import annotations

import json
from collections.abc import Iterable

from app.data.sources import load_sources


def build_model_card(
    *,
    base_model: str,
    repo_id: str,
    train_source_ids: Iterable[str],
    summary: dict[str, object],
) -> str:
    ids = set(train_source_ids)
    sources = [s for s in load_sources().sources if s.id in ids]
    has_notes = "notes" in ids
    has_nc = any(s.nc_flag or s.licence_class == "B" for s in sources)
    lines = [
        "---",
        f"base_model: {base_model}",
        "library_name: peft",
        "license: other",
        "tags: [qlora, lora, complex-systems, grounded-qa]",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"QLoRA adapter (nf4, r=16, alpha=32) for `{base_model}`: given a passage and a",
        "question, it answers in strict JSON with a verbatim quote, or abstains when",
        "the passage does not contain the answer. Trained on a free Kaggle T4.",
        "",
        "Source code and evaluation: https://github.com/WaiteM124/complex-systems-lora",
        "",
        "## Training sources and licences",
        "",
        "| Source | Title | Licence |",
        "|---|---|---|",
    ]
    for source in sorted(sources, key=lambda s: s.id):
        lines.append(f"| {source.id} | {source.title} | {source.licence or 'unknown'} |")
    if has_notes:
        lines.append("| notes | Author's own study notes (private, not published) | Author |")
    lines += [
        "",
        "Training items were generated synthetically from these passages; no passage",
        "text is published in this repository or with the adapter.",
    ]
    if has_nc:
        lines += [
            "",
            "**Non-commercial note:** some training sources are CC BY-NC-SA. Whether an",
            "adapter counts as an adaptation is unsettled, so use this adapter for",
            "non-commercial purposes only.",
        ]
    lines += ["", "## Training run", "", "```json"]
    lines.append(json.dumps(summary, indent=2, sort_keys=True, default=str))
    lines += ["```", ""]
    return "\n".join(lines)
