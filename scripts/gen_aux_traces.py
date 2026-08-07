"""Auxiliary (non-eval) trace generation for distillation training (plan todo 12).

Generates greedy traces from an AUXILIARY corpus that is disjoint from the
sealed evaluation manifest, so the distillation builder's leak-proof check
passes and training never sees eval prompts. Sources:

  math       - GSM8K TRAIN split (eval uses GSM8K test).
  code       - hand-authored completion prompts (not HumanEval/MBPP items).
  chat       - hand-authored chat prompts (not MT-Bench items).
  structured - hand-authored JSON-schema prompts (schemas differ from the
               sealed structured eval generator).

Writes runs/traces_aux/<task_class>.jsonl in the gen_traces.py schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

AUX_CODE = [
    "Write a Python function `fib(n)` returning the n-th Fibonacci number iteratively.",
    "Write a Python function `is_palindrome(s)` that checks a cleaned string both ways.",
    "Write a Python function `fizzbuzz(n)` returning the FizzBuzz sequence from 1 to n.",
    "Write a Python function `flatten(nested)` that flattens a nested list into one list.",
    "Write a Python function `word_count(text)` mapping each word to its frequency.",
    "Write a Python function `gcd(a, b)` using Euclid's algorithm.",
    "Write a Python function `binary_search(arr, target)` returning the index or -1.",
    "Write a Python function `reverse_words(s)` reversing the order of words.",
]

AUX_CHAT = [
    "Explain in two sentences how a transformer attention mechanism works.",
    "Give a short checklist for reviewing code before merging a pull request.",
    "Describe the difference between a compiler and an interpreter in one paragraph.",
    "Summarize the trade-offs between SQL and NoSQL databases briefly.",
    "What are three best practices for writing clear commit messages?",
    "Explain what an API rate limit is and why services use it.",
    "Describe how a hash table resolves collisions, in simple terms.",
    "What is the purpose of a virtual environment in Python development?",
]


def _schema(*pairs) -> dict:
    props = {name: {"type": typ} for name, typ in pairs}
    return {"type": "object", "properties": props, "required": list(props)}


AUX_STRUCTURED_SCHEMAS = [
    _schema(("name", "string"), ("age", "integer")),
    _schema(("city", "string"), ("population", "integer")),
    _schema(("product", "string"), ("price_usd", "number"), ("in_stock", "boolean")),
    _schema(("title", "string"), ("genre", "string")),
    _schema(("event_name", "string"), ("date", "string"), ("attendees", "integer")),
    _schema(("course", "string"), ("credits", "integer"), ("passed", "boolean")),
    _schema(("song", "string"), ("artist", "string"), ("year", "integer")),
    _schema(("recipe", "string"), ("servings", "integer"), ("vegetarian", "boolean")),
]


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _render(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _math_prompts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    take = min(n, len(ds))
    return [ds[i]["question"] for i in range(take)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out-root", default="runs/traces_aux")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("gen_aux_traces: CUDA GPU unavailable; refusing to run on CPU.", file=sys.stderr)
        return 2

    from familydraft.targets.wrapper import TargetModel

    campaign = _load_yaml(Path("configs/trace_campaign.yaml"))
    target = TargetModel.load(args.repo, dtype="bf16")

    aux_prompts = {
        "code": AUX_CODE[: args.per_class],
        "chat": AUX_CHAT[: args.per_class],
        "structured": [
            f"Produce a JSON object that satisfies this schema: {json.dumps(s)}"
            for s in AUX_STRUCTURED_SCHEMAS[: args.per_class]
        ],
        "math": _math_prompts(args.per_class),
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for task_class, prompts in aux_prompts.items():
        rows = []
        for idx, user_text in enumerate(prompts):
            prompt_text = _render(target.tokenizer, user_text)
            prompt_ids = target.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            torch.manual_seed(campaign["defaults"]["seed"])
            full = target.generate_greedy(prompt_ids, args.max_new_tokens)
            chosen = full[0, prompt_ids.shape[0] :].tolist()
            rows.append(
                {
                    "trace_id": f"aux-{task_class}-{idx}",
                    "task_class": task_class,
                    "dataset": "auxiliary",
                    "item_id": f"aux-{task_class}-{idx}",
                    "target_id": args.repo,
                    "seed": campaign["defaults"]["seed"],
                    "dtype": "bf16",
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_ids": prompt_ids.tolist(),
                    "chosen_tokens": chosen,
                    "config_sha256": "auxiliary",
                }
            )
        out_path = out_root / f"{task_class}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"[aux:{task_class}] wrote {len(rows)} traces -> {out_path}")

    print("gen_aux_traces: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
