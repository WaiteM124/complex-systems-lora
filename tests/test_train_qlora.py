from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from _synthetic import make_items, write_items
from app.data.reviewed import SplitLeakError
from app.model_config import get_base_model
from app.paths import repo_root
from app.task import target_json
from app.train import qlora
from app.train.config import attempt_ladder, load_train_config

GPU_MODULES = ("torch", "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets")


def test_recipe_matches_spec() -> None:
    cfg = load_train_config()
    assert cfg.model.load_in_4bit and cfg.model.bnb_4bit_quant_type == "nf4"
    assert cfg.model.max_seq_length == 2048
    assert (cfg.lora.r, cfg.lora.alpha) == (16, 32)
    assert set(cfg.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    t = cfg.training
    assert t.learning_rate == pytest.approx(2e-4)
    assert t.num_train_epochs == 3
    assert (t.per_device_train_batch_size, t.gradient_accumulation_steps) == (2, 8)
    assert t.gradient_checkpointing and t.fp16 and not t.bf16
    assert t.smoke_steps == 20
    assert cfg.output.working_dir == "/kaggle/working"
    assert cfg.output.adapter() == Path("/kaggle/working/adapter")


def test_oom_ladder() -> None:
    ladder = attempt_ladder(load_train_config())
    shapes = [
        (a.per_device_train_batch_size, a.gradient_accumulation_steps, a.max_seq_length)
        for a in ladder
    ]
    assert shapes == [(2, 8, 2048), (1, 16, 2048), (1, 16, 1536)]
    assert all(a.effective_batch == 16 for a in ladder)


def test_no_model_ids_hardcoded() -> None:
    base = get_base_model()
    assert base == "Qwen/Qwen3-4B-Instruct-2507"
    pattern = re.compile(r"Qwen/|qwen\.qwen3|deepseek\.")
    for path in [
        *(repo_root() / "src" / "app" / "train").glob("*.py"),
        *(repo_root() / "src" / "app" / "eval").glob("*.py"),
        repo_root() / "config" / "train.yaml",
        repo_root() / "config" / "eval.yaml",
    ]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_prepare_data_excludes_test(tmp_path: Path) -> None:
    path = write_items(tmp_path / "reviewed.jsonl", make_items(8, 4, 5))
    data = qlora.prepare_data(path, load_train_config())
    assert len(data.train) == 8
    assert all(item.split == "train" for item in data.train)
    assert len(data.val_subset) == 4
    assert len(data.test_split_sha256) == 64


def test_prepare_data_refuses_leak(tmp_path: Path) -> None:
    items = make_items(4, 2, 2)
    test_item = next(i for i in items if i.split == "test")
    leaked = test_item.model_copy(update={"id": "leaked", "split": "train"})
    path = write_items(tmp_path / "reviewed.jsonl", [*items, leaked])
    with pytest.raises(SplitLeakError):
        qlora.prepare_data(path, load_train_config())


def test_chat_records_end_with_target() -> None:
    items = make_items(3, 0, 0)
    records = qlora.chat_records(items, "SYS")
    assert all(r[0] == {"role": "system", "content": "SYS"} for r in records)
    assert [r[-1]["content"] for r in records] == [target_json(i.target) for i in items]


def test_val_subset_deterministic() -> None:
    val = [i for i in make_items(0, 20, 0)]
    a = qlora.select_val_subset(val, 5, seed=1)
    b = qlora.select_val_subset(list(reversed(val)), 5, seed=1)
    assert [i.id for i in a] == [i.id for i in b]
    assert len(qlora.select_val_subset(val, 50, seed=1)) == 20


def test_checkpoint_helpers(tmp_path: Path) -> None:
    for step in (100, 25, 50):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "other").mkdir()
    assert [p.name for p in qlora.list_checkpoints(tmp_path)] == [
        "checkpoint-25",
        "checkpoint-50",
        "checkpoint-100",
    ]
    final = tmp_path / "final"
    final.mkdir()
    (final / "adapter_model.safetensors").write_bytes(b"")
    names = [p.name for p in qlora.selection_candidates(tmp_path, 2)]
    assert names == ["checkpoint-50", "checkpoint-100", "final"]
    assert qlora.list_checkpoints(tmp_path / "missing") == []


def test_pick_best_prefers_earlier_on_tie() -> None:
    assert qlora.pick_best({"a": 0.5, "b": 0.7, "c": 0.7}) == "b"
    assert qlora.pick_best({"a": None, "b": 0.1}) == "b"
    with pytest.raises(ValueError):
        qlora.pick_best({})


class OutOfMemoryError(RuntimeError):
    pass


def test_oom_detection_and_rungs() -> None:
    assert qlora.is_cuda_oom(OutOfMemoryError("x"))
    assert qlora.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not qlora.is_cuda_oom(ValueError("bad"))
    assert qlora.start_rung({}, None, 3) == 0
    assert qlora.start_rung({"rung": 2}, None, 3) == 2
    assert qlora.start_rung({"rung": 2}, 1, 3) == 1
    with pytest.raises(ValueError):
        qlora.start_rung({}, 5, 3)


def test_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    assert qlora.read_state(path) == {}
    qlora.write_state(path, {"rung": 1})
    assert qlora.read_state(path) == {"rung": 1}


def test_hf_token_from_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "  hf_dummy_value  ")
    assert qlora.hf_token() == "hf_dummy_value"
    monkeypatch.delenv("HF_TOKEN")
    assert qlora.hf_token() is None  # no kaggle_secrets module off Kaggle


def test_warmup_steps() -> None:
    attempt = attempt_ladder(load_train_config())[0]
    # 1,320 examples / 16 = 83 steps/epoch -> 249 total -> 3% = 7.47 -> 8
    assert qlora.warmup_steps(1320, attempt, 3, 0.03) == 8


def test_check_data_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_items(tmp_path / "reviewed.jsonl", make_items(8, 4, 5))
    assert qlora.main(["--data", str(path), "--check-data"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"] == {"train": 8, "val": 4, "test": 5}
    assert out["no_test_in_train"] is True


def test_modules_do_not_import_gpu_libraries() -> None:
    code = (
        "import sys\n"
        "import app.train.qlora, app.train.model_card, app.eval.local, app.eval.judge\n"
        "import app.eval.bedrock_baseline, app.eval.score, app.eval.gate\n"
        f"bad = [m for m in {GPU_MODULES!r} if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"APP_ENV": "local"}
    )
    assert result.returncode == 0, result.stderr


def test_model_card_lists_sources_without_text() -> None:
    from app.train.model_card import build_model_card

    card = build_model_card(
        base_model="base/model",
        repo_id="owner/repo",
        train_source_ids={"cs-018", "cs-003", "notes"},
        summary={"n_train": 1},
    )
    assert "cs-018" in card and "cs-003" in card
    assert "Non-commercial" in card
    assert "study notes" in card
