from __future__ import annotations

import ast
import json
import tomllib
from typing import Any

import pytest

from app.paths import repo_root

NOTEBOOKS = ("train_qlora.ipynb", "eval.ipynb")


def _load(name: str) -> dict[str, Any]:
    path = repo_root() / "notebooks" / name
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _code(nb: dict[str, Any]) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_outputs_and_parses(name: str) -> None:
    nb = _load(name)
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
    for source in _code(nb):
        ast.parse(source)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_installs_pinned_extra_and_hides_token(name: str) -> None:
    text = "\n".join(_code(_load(name)))
    assert "[train]" in text
    assert 'CUDA_VISIBLE_DEVICES"] = "0"' in text
    for line in text.splitlines():
        assert not ("print" in line and 'HF_TOKEN"]' in line)


def test_train_notebook_runs_smoke_then_full() -> None:
    text = "\n".join(_code(_load("train_qlora.ipynb")))
    smoke = text.index('"app.train.qlora", "--config", "config/train.yaml", "--smoke"')
    assert smoke < text.rindex('"app.train.qlora", "--config", "config/train.yaml")')
    assert "peak_vram_reserved_gib" in text


def test_eval_notebook_runs_both_systems_and_scores() -> None:
    text = "\n".join(_code(_load("eval.ipynb")))
    assert '"--system", "base"' in text
    assert '"--system", "lora"' in text
    assert '"app.eval.score", "--kaggle"' in text


def test_train_extra_is_fully_pinned() -> None:
    raw = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    train = raw["project"]["optional-dependencies"]["train"]
    assert train and all("==" in req for req in train)
    names = {req.split("==")[0] for req in train}
    assert {"unsloth", "trl", "peft", "transformers", "bitsandbytes"} <= names
    dev = raw["project"]["optional-dependencies"]["dev"]
    assert not any(req.split("==")[0].split(">")[0] in names for req in dev)


def test_notebooks_dir_has_only_expected_files() -> None:
    files = sorted(p.name for p in (repo_root() / "notebooks").iterdir() if p.is_file())
    assert files == sorted(NOTEBOOKS)
