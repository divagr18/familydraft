"""Integrated speculative speedup evaluation (plan Wave E / todo 22 local scale).

Measures wall-clock vanilla greedy decoding vs the integrated speculative loop
on code prompts, asserts the speculative output is token-identical to vanilla
(losslessness), and reports speedup + acceptance. Runs on the local GPU at
Qwen3-0.6B scale; the registered campaign uses bigger targets on RunPod.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

CODE_PROMPTS = [
    "Write a Python function that returns the sum of a list of numbers:",
    "Write 8 Python lines of the form value_i = i * i for i from 0 to 7:",
    "Write a Python function to compute the factorial of n using a loop:",
    "Write 6 dictionary entries mapping 'key_i' to i for i from 0 to 5:",
    "Write a Python function that reverses a string using a loop:",
    "Write 8 list append statements adding i to a list for i from 0 to 7:",
]


def _gpu_time(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--drafters", default="copy,general")
    parser.add_argument("--out", default="runs/results/integrated_speedup.json")
    args = parser.parse_args()

    from familydraft.draft.trunk import build_trunk_from_config
    from familydraft.eval.draft_loop import (
        IntegratedSpeculator,
        make_copy_drafter,
        make_general_drafter,
    )
    from familydraft.experts.copy import CopyExpert
    from familydraft.experts.general import GeneralExpert
    from familydraft.targets.wrapper import TargetModel

    if not torch.cuda.is_available():
        print("run_speculative_eval: CUDA GPU unavailable.", flush=True)
        return 2

    device = torch.device("cuda")
    target = TargetModel.load(args.repo, dtype="bf16")

    drafters = {}
    copy_expert = CopyExpert(seed=4, min_length=3)
    if "copy" in args.drafters:
        drafters["copy"] = lambda: make_copy_drafter(copy_expert, args.spec_len)
    if "general" in args.drafters:
        trunk = build_trunk_from_config(Path("."))
        expert = GeneralExpert(trunk).to(device)
        drafters["general"] = lambda: make_general_drafter(expert, args.spec_len, 0, device)

    # Warmup
    warm_prompt = target.tokenizer("def f():", return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    target.generate_greedy(warm_prompt, 8)

    results = {"schema": "familydraft.integrated_speedup.v1", "repo": args.repo,
               "max_new_tokens": args.max_new, "spec_len": args.spec_len, "runs": []}

    for prompt_text in CODE_PROMPTS:
        prompt_ids = target.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        prompt_list = prompt_ids.tolist()

        vanilla_time = _gpu_time(lambda: target.generate_greedy(prompt_ids, args.max_new))
        vanilla_tokens = target.generate_greedy(prompt_ids, args.max_new)[
            0, len(prompt_list):
        ].tolist()

        row = {"prompt": prompt_text[:60], "vanilla_seconds": vanilla_time,
               "tokens": len(vanilla_tokens)}

        for name, factory in drafters.items():
            speculator = IntegratedSpeculator(target, factory(), args.spec_len, 0)
            spec_time = _gpu_time(lambda: speculator.generate(prompt_list, args.max_new))
            res = speculator.generate(prompt_list, args.max_new)
            agree = sum(1 for a, b in zip(res["tokens"], vanilla_tokens) if a == b)
            row[f"{name}_seconds"] = spec_time
            row[f"{name}_speedup"] = vanilla_time / max(spec_time, 1e-9)
            row[f"{name}_tokens_per_round"] = res["tokens_per_round"]
            row[f"{name}_agreement"] = agree / max(1, len(vanilla_tokens))
        results["runs"].append(row)

    def _avg(key):
        vals = [r[key] for r in results["runs"] if key in r]
        return sum(vals) / max(1, len(vals))

    results["summary"] = {}
    for name in drafters:
        results["summary"][name] = {
            "mean_speedup": _avg(f"{name}_speedup"),
            "mean_tokens_per_round": _avg(f"{name}_tokens_per_round"),
            "mean_agreement": _avg(f"{name}_agreement"),
        }
    results["summary"]["vanilla_ms_per_token"] = (
        1000.0 * sum(r["vanilla_seconds"] for r in results["runs"])
        / max(1, sum(r["tokens"] for r in results["runs"]))
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
