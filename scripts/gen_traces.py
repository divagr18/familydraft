"""Greedy trace generation (plan todo 10).

Generates greedy continuations for evaluation prompts and records the
sequences used by the oracle analysis (todo 11) and the distillation
dataset builder (todo 12). On RunPod this script runs against the full
target matrix with top-k logit capture; locally it is capped by the
local_run limits pinned in configs/trace_campaign.yaml.

Output: one JSONL file per task class at runs/traces/<model_name>/<class>.jsonl
with lines:
  {trace_id, task_class, dataset, item_id, target_id, seed, dtype,
   max_new_tokens, prompt_ids, chosen_tokens, config_sha256}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from familydraft.infra.run import config_fingerprint


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_eval_classes() -> dict[str, list[str]]:
    thresholds = _load_yaml(Path("configs/oracle_thresholds.yaml"))
    return thresholds["classes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--classes", default="chat,code,math,structured")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out-root", default="runs/traces")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("gen_traces: CUDA GPU unavailable; refusing to run on CPU.", file=sys.stderr)
        return 2

    from familydraft.targets.wrapper import TargetModel

    campaign = _load_yaml(Path("configs/trace_campaign.yaml"))
    class_map = _load_eval_classes()
    requested = [c.strip() for c in args.classes.split(",")]
    unknown = [c for c in requested if c not in class_map]
    if unknown:
        print(f"gen_traces: unknown classes {unknown}; known={sorted(class_map)}", file=sys.stderr)
        return 2

    max_prompts = args.max_prompts or campaign["local_run"]["max_prompts_per_class"]
    default_tokens = args.max_new_tokens or campaign["local_run"]["max_new_tokens"]

    config = {
        "repo": args.repo,
        "classes": requested,
        "max_prompts": max_prompts,
        "default_tokens": default_tokens,
        "seed": campaign["defaults"]["seed"],
    }
    config_sha = config_fingerprint(config)

    target = TargetModel.load(args.repo, dtype=campaign["defaults"]["dtype"])
    model_name = args.repo.split("/")[-1]
    out_dir = Path(args.out_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_class in requested:
        datasets = class_map[task_class]
        class_budget = max_prompts
        max_new = args.max_new_tokens or min(
            default_tokens, campaign["max_new_tokens_by_class"][task_class]
        )
        rows: list[dict] = []
        for dataset in datasets:
            items_path = Path("data/eval") / dataset / "items.jsonl"
            with items_path.open(encoding="utf-8") as handle:
                items = [json.loads(line) for line in handle if line.strip()]
            take = min(len(items), class_budget)
            class_budget -= take
            for item in items[:take]:
                prompt_ids = target.tokenizer(
                    item["prompt_text"], return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                torch.manual_seed(campaign["defaults"]["seed"])
                full = target.generate_greedy(prompt_ids, max_new)
                chosen = full[0, prompt_ids.shape[0] :].tolist()
                rows.append(
                    {
                        "trace_id": f"{dataset}:{item['id']}",
                        "task_class": task_class,
                        "dataset": dataset,
                        "item_id": item["id"],
                        "target_id": args.repo,
                        "seed": campaign["defaults"]["seed"],
                        "dtype": campaign["defaults"]["dtype"],
                        "max_new_tokens": max_new,
                        "prompt_ids": prompt_ids.tolist(),
                        "chosen_tokens": chosen,
                        "config_sha256": config_sha,
                    }
                )
                print(f"  [{task_class}] {dataset}:{item['id']} -> {len(chosen)} tokens")
            if class_budget <= 0:
                break
        out_path = out_dir / f"{task_class}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"[{task_class}] wrote {len(rows)} traces -> {out_path}")

    print(f"gen_traces: done, config_sha256={config_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
