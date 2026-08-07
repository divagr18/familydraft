"""Generate distillation training traces from a target repo (plan todo 10/15).

Runs the target's greedy decoding over a diverse TRAINING corpus (disjoint from
the sealed eval set) and records prompt_ids + chosen_tokens in the same schema
as the auxiliary traces, so build_distill_dataset.py can build shards from it.

Usage (on RunPod for 8B):
  python scripts/gen_train_data.py --repo Qwen/Qwen3-8B --per-class 40 \
      --max-new 96 --out-dir runs/traces_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

_CODE_SUBJECTS = ["numbers", "strings", "words", "items", "values", "scores",
                  "names", "prices", "temps", "weights"]
_CODE_TEMPLATES = [
    "Write a Python function that sums a list of {s}.",
    "Write a Python function that reverses a list of {s}.",
    "Write a Python function that filters the even elements of a list of {s}.",
    "Write a Python function that finds the maximum of a list of {s}.",
    "Write a Python function that counts the elements in a list of {s}.",
    "Write a Python function that sorts a list of {s} in descending order.",
    "Write a Python function that removes duplicates from a list of {s}.",
    "Write a Python function that computes the average of a list of {s}.",
]

_CHAT_TOPICS = ["photosynthesis", "neural networks", "the water cycle", "gravity",
                "supply and demand", "compound interest", "the immune system",
                "plate tectonics", "machine learning", "the greenhouse effect"]
_CHAT_TEMPLATES = [
    "Explain {t} in two sentences.",
    "What is {t} and why does it matter?",
    "Give a brief overview of {t}.",
    "Summarize the key idea behind {t}.",
]

_STRUCTURED_SCHEMAS = [
    ('{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},'
     '"required":["name","age"]}', "a person"),
    ('{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},'
     '"required":["city","population"]}', "a city"),
    ('{"type":"object","properties":{"product":{"type":"string"},"price":{"type":"number"}},'
     '"required":["product","price"]}', "a product"),
    ('{"type":"object","properties":{"title":{"type":"string"},"year":{"type":"integer"}},'
     '"required":["title","year"]}', "a book"),
    ('{"type":"object","properties":{"course":{"type":"string"},"credits":{"type":"integer"}},'
     '"required":["course","credits"]}', "a university course"),
    ('{"type":"object","properties":{"song":{"type":"string"},"artist":{"type":"string"}},'
     '"required":["song","artist"]}', "a song"),
]


def _code_prompts(n: int) -> list[str]:
    out = [t.format(s=s) for t in _CODE_TEMPLATES for s in _CODE_SUBJECTS]
    return out[:n]


def _chat_prompts(n: int) -> list[str]:
    out = [t.format(t=topic) for t in _CHAT_TEMPLATES for topic in _CHAT_TOPICS]
    return out[:n]


def _structured_prompts(n: int) -> list[str]:
    out = [f"Produce a JSON object that satisfies this schema: {sch} for {what}."
           for sch, what in _STRUCTURED_SCHEMAS]
    return out[:n]


def _math_prompts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    take = min(n, len(ds))
    return [ds[i]["question"] for i in range(take)]


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--out-dir", default="runs/traces_train")
    parser.add_argument("--classes", default="code,chat,structured,math")
    args = parser.parse_args()

    from familydraft.targets.wrapper import TargetModel

    if not torch.cuda.is_available():
        print("gen_train_data: CUDA GPU unavailable.", flush=True)
        return 2

    campaign = _load_yaml(Path("configs/trace_campaign.yaml"))
    target = TargetModel.load(args.repo, dtype="bf16")
    target_id = args.repo

    builders = {
        "code": _code_prompts,
        "chat": _chat_prompts,
        "structured": _structured_prompts,
        "math": _math_prompts,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_class in args.classes.split(","):
        task_class = task_class.strip()
        if task_class not in builders:
            continue
        prompts = builders[task_class](args.per_class)
        rows = []
        for idx, user_text in enumerate(prompts):
            prompt_text = target.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = target.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            torch.manual_seed(campaign["defaults"]["seed"])
            full = target.generate_greedy(prompt_ids, args.max_new)
            chosen = full[0, prompt_ids.shape[0]:].tolist()
            rows.append({
                "trace_id": f"train-{task_class}-{idx}",
                "task_class": task_class,
                "dataset": "train-gen",
                "item_id": f"train-{task_class}-{idx}",
                "target_id": target_id,
                "seed": campaign["defaults"]["seed"],
                "dtype": "bf16",
                "max_new_tokens": args.max_new,
                "prompt_ids": prompt_ids.tolist(),
                "chosen_tokens": chosen,
                "config_sha256": "train-gen",
            })
        out_path = out_dir / f"{task_class}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"[train:{task_class}] wrote {len(rows)} traces -> {out_path}", flush=True)

    print("gen_train_data: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
