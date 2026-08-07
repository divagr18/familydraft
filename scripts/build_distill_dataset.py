"""Target-ID-tagged distillation dataset builder with LEAK PROOF (plan todo 12).

Converts auxiliary traces into training shards `data/distill/<split>/shard-*.arrow`
with records (input_ids, target_ids, prompt_len, target_id, task_class,
topk_ids, topk_logits). Guarantees:

  * Every record's target_id is in configs/target_ids.json (else refuse).
  * LEAK PROOF: a record whose canonical prompt hash matches ANY sealed
    eval-manifest prompt is tagged split=diag and refused for training
    shards. Eval prompts may appear in traces for oracle diagnostics only.

Canonical prompt hash = sha256 of the tokenized prompt-id tuple, so eval and
trace prompts are compared in the same representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical_hash(prompt_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(prompt_ids)).encode("utf-8")).hexdigest()


def build_eval_hash_set(manifest_path: Path, tokenizer) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: set[str] = set()
    for ds_info in manifest.get("datasets", {}).values():
        data_file = manifest_path.parent / ds_info["data_file"]
        with data_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                ids = tokenizer(
                    item["prompt_text"], return_tensors="pt", add_special_tokens=False
                )["input_ids"][0].tolist()
                hashes.add(_canonical_hash(ids))
    return hashes


def _target_table(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_schema() -> pa.Schema:
    return pa.schema(
        [
            ("input_ids", pa.list_(pa.int64())),
            ("target_ids", pa.list_(pa.int64())),
            ("prompt_len", pa.int64()),
            ("target_id", pa.int32()),
            ("task_class", pa.string()),
            ("topk_ids", pa.list_(pa.list_(pa.int64()))),
            ("topk_logits", pa.list_(pa.list_(pa.float32()))),
        ]
    )


def _to_arrow_batch(records: list[dict]) -> pa.RecordBatch:
    schema = _record_schema()
    arrays = {
        "input_ids": [r["input_ids"] for r in records],
        "target_ids": [r["target_ids"] for r in records],
        "prompt_len": [r["prompt_len"] for r in records],
        "target_id": [r["target_id"] for r in records],
        "task_class": [r["task_class"] for r in records],
        "topk_ids": [r.get("topk_ids") for r in records],
        "topk_logits": [r.get("topk_logits") for r in records],
    }
    return pa.record_batch([pa.array(arrays[f.name], type=f.type) for f in schema], schema=schema)


def filter_traces(
    rows: list[dict], eval_hashes: set[str], target_table: dict
) -> tuple[list[dict], int]:
    """Split trace rows into training records vs eval-diagnostic exclusions.

    Raises ValueError naming the offending record for an unknown target_id.
    Returns (training_records, diag_excluded_count).
    """
    records: list[dict] = []
    diag_excluded = 0
    for offset, row in enumerate(rows):
        target_id = row.get("target_id")
        if target_id not in target_table:
            raise ValueError(
                f"unknown target_id {target_id!r} at record offset {offset}"
            )
        prompt_ids = row["prompt_ids"]
        chosen = row["chosen_tokens"]
        if _canonical_hash(prompt_ids) in eval_hashes:
            diag_excluded += 1
            continue
        records.append(
            {
                "input_ids": list(prompt_ids) + list(chosen[:-1]),
                "target_ids": list(chosen),
                "prompt_len": len(prompt_ids),
                "target_id": target_table[target_id]["id"],
                "task_class": row["task_class"],
                "topk_ids": row.get("topk_ids"),
                "topk_logits": row.get("topk_logits"),
            }
        )
    return records, diag_excluded


def _read_trace_rows(traces_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for tf in sorted(traces_dir.glob("*.jsonl")):
        with tf.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", default="runs/traces_aux")
    parser.add_argument("--out-root", default="data/distill")
    parser.add_argument("--repo-for-tokenizer", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    distill_cfg = _load_yaml(Path("configs/distill.yaml"))
    target_table = _target_table(Path("configs/target_ids.json"))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.repo_for_tokenizer)
    eval_hashes = build_eval_hash_set(Path(distill_cfg["eval_manifest"]), tokenizer)
    print(f"eval hash set size: {len(eval_hashes)}")

    trace_files = sorted(Path(args.traces_dir).glob("*.jsonl"))
    if not trace_files:
        print(f"build_distill_dataset: no traces in {args.traces_dir}", file=sys.stderr)
        return 2

    rows = _read_trace_rows(Path(args.traces_dir))
    try:
        records, diag_excluded = filter_traces(rows, eval_hashes, target_table)
    except ValueError as exc:
        print(f"build_distill_dataset: {exc}", file=sys.stderr)
        return 3

    # Defensive re-check: no training record may collide with an eval prompt.
    for r in records:
        if _canonical_hash(r["input_ids"][: r["prompt_len"]]) in eval_hashes:
            print("build_distill_dataset: LEAK detected after filtering", file=sys.stderr)
            return 4

    out_root = Path(args.out_root) / "train"
    out_root.mkdir(parents=True, exist_ok=True)
    shard_max = distill_cfg["shard_max_records"]
    # Stratify by task_class then shard; preserve insertion order per class.
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r["task_class"], []).append(r)
    ordered: list[dict] = []
    for task_class in sorted(by_class):
        ordered.extend(by_class[task_class])

    shard_idx = 0
    for start in range(0, len(ordered), shard_max):
        batch_records = ordered[start : start + shard_max]
        batch = _to_arrow_batch(batch_records)
        out_path = out_root / f"shard-{shard_idx:04d}.arrow"
        with out_path.open("wb") as sink:
            with pa.ipc.new_file(sink, batch.schema) as writer:
                writer.write_batch(batch)
        shard_idx += 1

    # Stratification report
    total = max(1, len(records))
    ratios = {tc: len(by_class.get(tc, [])) / total for tc in distill_cfg["ratio_by_task_class"]}
    summary = {
        "records": len(records),
        "diag_excluded": diag_excluded,
        "shards": shard_idx,
        "ratio_by_task_class": ratios,
        "configured_ratio": distill_cfg["ratio_by_task_class"],
    }
    (Path(args.out_root) / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
