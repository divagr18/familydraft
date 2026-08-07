"""Generate distillation training traces from a target repo (plan todo 10/15).

Runs the target's greedy decoding over a diverse TRAINING corpus (disjoint from
the sealed eval set) and records prompt_ids + chosen_tokens for
build_distill_dataset.py.

Speedups over the v1 single-pass generator:
  * batched greedy decode (--batch prompts per model call),
  * parallel model instances (--workers processes; each loads its own copy of
    the target and shards the prompt list - requires VRAM for N instances).

Usage (on RunPod for 8B, A100-80GB, 2 workers x batch 8):
  python scripts/gen_train_data.py --repo Qwen/Qwen3-8B --per-class 100 \
      --max-new 128 --workers 2 --batch 8 --out-dir runs/traces_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

_CODE_SUBJECTS = ["numbers", "strings", "words", "items", "values", "scores",
                  "names", "prices", "temps", "weights", "grades", "ages",
                  "lengths", "colors", "shapes", "emails", "ids", "dates",
                  "sizes", "counts"]
_CODE_TEMPLATES = [
    "Write a Python function that sums a list of {s}.",
    "Write a Python function that reverses a list of {s}.",
    "Write a Python function that filters the even elements of a list of {s}.",
    "Write a Python function that finds the maximum of a list of {s}.",
    "Write a Python function that counts the elements in a list of {s}.",
    "Write a Python function that sorts a list of {s} in descending order.",
    "Write a Python function that removes duplicates from a list of {s}.",
    "Write a Python function that computes the average of a list of {s}.",
    "Write a Python function that returns the median of a list of {s}.",
    "Write a Python function that maps each {s} to its length.",
    "Write a Python function that concatenates two lists of {s}.",
    "Write a Python function that returns the first five elements of a list of {s}.",
    "Write a Python function that pairs two lists of {s} into tuples.",
    "Write a Python function that checks whether a list of {s} is sorted.",
    "Write a Python function that returns the n largest elements of a list of {s}.",
    "Write a Python function that swaps adjacent elements of a list of {s}.",
]

_CHAT_TOPICS = ["photosynthesis", "neural networks", "the water cycle", "gravity",
                "supply and demand", "compound interest", "the immune system",
                "plate tectonics", "machine learning", "the greenhouse effect",
                "biodiversity", "quantum computing", "renewable energy",
                "the scientific method", "game theory", "cybersecurity",
                "urban planning", "genetics", "climate models", "cryptography",
                "blockchain", "regression analysis", "the periodic table",
                "information theory", "linguistics", "electromagnetism",
                "statistical inference", "organizational behavior",
                "neuroplasticity", "aerodynamics"]
_CHAT_TEMPLATES = [
    "Explain {t} in two sentences.",
    "What is {t} and why does it matter?",
    "Give a brief overview of {t}.",
    "Summarize the key idea behind {t}.",
    "List three important facts about {t}.",
    "Describe how {t} relates to everyday life.",
    "Compare {t} with its opposite concept.",
    "Write a one-paragraph definition of {t}.",
    "Explain {t} to a ten-year-old.",
    "Discuss a common misconception about {t}.",
]

_KEY_TYPES = [("name", "string"), ("age", "integer"), ("city", "string"),
              ("price", "number"), ("title", "string"), ("year", "integer"),
              ("enabled", "boolean"), ("score", "number"), ("color", "string"),
              ("weight", "number"), ("count", "integer"), ("email", "string")]
_CONTEXTS = ["a product", "a person", "a city", "an event", "a book", "a recipe",
             "an order", "a sensor reading", "a course", "a song", "a building",
             "a vehicle", "a plant", "an animal", "a team", "a movie"]


def _code_prompts(n: int) -> list[str]:
    return [t.format(s=s) for t in _CODE_TEMPLATES for s in _CODE_SUBJECTS][:n]


def _chat_prompts(n: int) -> list[str]:
    return [t.format(t=topic) for t in _CHAT_TEMPLATES for topic in _CHAT_TOPICS][:n]


def _structured_prompts(n: int) -> list[str]:
    import random

    rng = random.Random(7)
    out = []
    for i in range(n):
        k = 2 + (i % 3)
        chosen = rng.sample(_KEY_TYPES, k)
        props = {key: {"type": typ} for key, typ in chosen}
        schema = json.dumps(
            {"type": "object", "properties": props, "required": list(props)}
        )
        context = _CONTEXTS[i % len(_CONTEXTS)]
        out.append(
            f"Produce a JSON object that satisfies this schema: {schema} for {context}."
        )
    return out


def _math_prompts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    take = min(n, len(ds))
    return [ds[i]["question"] for i in range(take)]


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_tasks(args) -> list[tuple[str, str]]:
    builders = {
        "code": _code_prompts,
        "chat": _chat_prompts,
        "structured": _structured_prompts,
        "math": _math_prompts,
    }
    tasks: list[tuple[str, str]] = []
    for task_class in args.classes.split(","):
        task_class = task_class.strip()
        if task_class not in builders:
            continue
        prompts = builders[task_class](args.per_class)
        tasks.extend((task_class, text) for text in prompts)
    return tasks


def _worker_main(
    task_chunk: list[tuple[str, str]], worker_idx: int, num_workers: int, args
) -> int:
    """Load the target and generate traces for one chunk; write worker_<i>.jsonl."""
    from familydraft.targets.wrapper import TargetModel

    if not torch.cuda.is_available():
        print("gen_train_data: CUDA GPU unavailable.", flush=True)
        return 2
    num_devices = torch.cuda.device_count()
    device = worker_idx % max(1, num_devices)
    torch.cuda.set_device(device)

    campaign = _load_yaml(Path("configs/trace_campaign.yaml"))
    target = TargetModel.load(args.repo, dtype="bf16")
    max_new = args.max_new

    by_class: dict[str, list[str]] = {}
    for task_class, text in task_chunk:
        by_class.setdefault(task_class, []).append(text)

    out_path = Path(args.out_dir) / f"worker_{worker_idx}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    batch = max(1, args.batch)
    for task_class, texts in by_class.items():
        for start in range(0, len(texts), batch):
            chunk_texts = texts[start : start + batch]
            prompt_texts = [
                target.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for text in chunk_texts
            ]
            prompts = [
                target.tokenizer(pt, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ][0].tolist()
                for pt in prompt_texts
            ]
            torch.manual_seed(campaign["defaults"]["seed"])
            continuations = target.generate_greedy_batch(prompts, max_new)
            for text, pids, chosen in zip(chunk_texts, prompts, continuations):
                if not chosen:
                    continue
                rows.append({
                    "trace_id": f"train-{task_class}-{worker_idx}-{len(rows)}",
                    "task_class": task_class,
                    "dataset": "train-gen",
                    "item_id": f"train-{task_class}-{worker_idx}-{len(rows)}",
                    "target_id": args.repo,
                    "seed": campaign["defaults"]["seed"],
                    "dtype": "bf16",
                    "max_new_tokens": max_new,
                    "prompt_ids": pids,
                    "chosen_tokens": chosen,
                    "config_sha256": "train-gen",
                })
        print(f"[worker {worker_idx}] {task_class}: {len(rows)} rows so far", flush=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[worker {worker_idx}] wrote {len(rows)} rows -> {out_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--out-dir", default="runs/traces_train")
    parser.add_argument("--classes", default="code,chat,structured,math")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(args)
    print(
        f"total training tasks: {len(tasks)} "
        f"(workers={args.workers}, batch={args.batch})",
        flush=True,
    )

    if args.workers <= 1:
        return _worker_main(tasks, 0, 1, args)

    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    chunk_size = max(1, len(tasks) // args.workers)
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)][
        : args.workers
    ]
    procs = [
        ctx.Process(target=_worker_main, args=(chunk, i, args.workers, args))
        for i, chunk in enumerate(chunks)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    if any(c != 0 for c in codes):
        print(f"gen_train_data: worker failures: {codes}", flush=True)
        return 1
    print("gen_train_data: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
