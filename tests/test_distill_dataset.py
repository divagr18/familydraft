"""Leak-proof distillation dataset tests (plan todo 12).

The central guarantee under test: NO training shard may contain a record whose
prompt matches any sealed evaluation-manifest prompt (eval leakage would make
every downstream speedup number invalid).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pyarrow.ipc as ipc
import pytest

REPO_ROOT = Path(__file__).parent.parent
DISTILL_ROOT = REPO_ROOT / "data" / "distill"
TRAIN_DIR = DISTILL_ROOT / "train"

_spec = importlib.util.spec_from_file_location(
    "build_distill_dataset", REPO_ROOT / "scripts" / "build_distill_dataset.py"
)
builder = importlib.util.module_from_spec(_spec)
sys.modules["build_distill_dataset"] = builder
_spec.loader.exec_module(builder)


def _canonical_hash(prompt_ids) -> str:
    return hashlib.sha256(json.dumps(list(prompt_ids)).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def eval_hashes(tokenizer):
    return builder.build_eval_hash_set(REPO_ROOT / "data" / "eval" / "MANIFEST.json", tokenizer)


@pytest.fixture(scope="module")
def target_table():
    return json.loads((REPO_ROOT / "configs" / "target_ids.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shard_records():
    records = []
    for shard in sorted(TRAIN_DIR.glob("shard-*.arrow")):
        with ipc.open_file(shard) as reader:
            table = reader.read_all()
        for i in range(table.num_rows):
            records.append(
                {
                    "input_ids": table["input_ids"][i].as_py(),
                    "target_ids": table["target_ids"][i].as_py(),
                    "prompt_len": table["prompt_len"][i].as_py(),
                    "target_id": table["target_id"][i].as_py(),
                    "task_class": table["task_class"][i].as_py(),
                }
            )
    return records


def test_shards_exist_and_round_trip_shapes(shard_records) -> None:
    assert shard_records, "no training shard found"
    for r in shard_records:
        assert len(r["target_ids"]) > 0
        assert r["prompt_len"] > 0
        assert len(r["input_ids"]) == r["prompt_len"] + len(r["target_ids"]) - 1


def test_target_ids_are_known(shard_records, target_table) -> None:
    known = {info["id"] for info in target_table.values()}
    for r in shard_records:
        assert r["target_id"] in known


def test_leak_proof_no_eval_prompt_in_training_shards(shard_records, eval_hashes) -> None:
    for r in shard_records:
        prompt = r["input_ids"][: r["prompt_len"]]
        assert _canonical_hash(prompt) not in eval_hashes, "eval prompt leaked into training"


def test_stratification_within_tolerance() -> None:
    summary = json.loads((DISTILL_ROOT / "build_summary.json").read_text(encoding="utf-8"))
    configured = summary["configured_ratio"]
    actual = summary["ratio_by_task_class"]
    for task_class, target_ratio in configured.items():
        assert abs(actual[task_class] - target_ratio) <= 0.05, (
            f"{task_class} ratio {actual[task_class]} deviates from {target_ratio}"
        )


def test_injected_eval_prompt_is_refused(tokenizer, eval_hashes, target_table) -> None:
    with (REPO_ROOT / "data" / "eval" / "mtbench" / "items.jsonl").open(encoding="utf-8") as f:
        eval_item = json.loads(f.readline())
    eval_prompt_ids = tokenizer(
        eval_item["prompt_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"][0].tolist()
    poisoned = {
        "task_class": "chat",
        "target_id": "Qwen/Qwen3-0.6B",
        "prompt_ids": eval_prompt_ids,
        "chosen_tokens": [1, 2, 3],
    }
    records, diag = builder.filter_traces([poisoned], eval_hashes, target_table)
    assert records == []
    assert diag == 1


def test_unknown_target_id_is_refused(eval_hashes, target_table) -> None:
    bad = {
        "task_class": "chat",
        "target_id": "NotAReal/Model",
        "prompt_ids": [9, 9, 9],
        "chosen_tokens": [1, 2, 3],
    }
    with pytest.raises(ValueError, match="unknown target_id"):
        builder.filter_traces([bad], eval_hashes, target_table)


def test_clean_aux_record_is_accepted(eval_hashes, target_table) -> None:
    clean = {
        "task_class": "code",
        "target_id": "Qwen/Qwen3-0.6B",
        "prompt_ids": [100001, 100002, 100003],
        "chosen_tokens": [5, 6, 7, 8],
    }
    records, diag = builder.filter_traces([clean], eval_hashes, target_table)
    assert diag == 0
    assert len(records) == 1
    assert records[0]["target_id"] == target_table["Qwen/Qwen3-0.6B"]["id"]
