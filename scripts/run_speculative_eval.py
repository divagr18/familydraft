"""Integrated speculative speedup evaluation (plan Wave E / todo 22).

Measures wall-clock vanilla greedy decoding vs the integrated speculative loop,
reports speedup + acceptance + agreement. Prompt sets:
  code        - general coding prompts (copy has moderate value)
  repetitive  - prompts engineered to elicit highly repetitive output, where the
                copy drafter should earn a real speedup
  structured  - JSON schema / repeated-key generation

Can load a trained general-expert checkpoint via --general-checkpoint.
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

REPETITIVE_PROMPTS = [
    "Write exactly 10 lines. Line i must be exactly: x_i = i * 3  (i from 0 to 9).",
    "Write 10 statements of the form results.append(i), one per line, for i from 0 to 9.",
    "Repeat the exact line print(i) ten times, once per line.",
    "Write 8 Python assignments of the form a_i = i for i from 0 to 7, one per line.",
    "Generate a JSON array of 8 objects. Each object has exactly the keys \"id\", "
    "\"name\", \"value\". The id runs from 0 to 7.",
    "Create a JSON object with keys item_0 through item_7, where each key maps to "
    "its index as the value.",
    "Write a JSON list of 8 strings. The k-th string must be \"entry_k\" for k 0 to 7.",
    "Output a CSV with header id,name,score followed by 8 rows whose id runs 0 to 7.",
]

STRUCTURED_PROMPTS = [
    "Produce a JSON object matching the schema {\"type\":\"object\",\"properties\":"
    "{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}},\"required\":"
    "[\"name\",\"age\"]} for a person named Ada aged 36.",
    "Produce a JSON object matching the schema {\"type\":\"object\",\"properties\":"
    "{\"city\":{\"type\":\"string\"},\"population\":{\"type\":\"integer\"}},\"required\":"
    "[\"city\",\"population\"]} for Tokyo.",
    "Produce a JSON array of 6 objects, each with keys \"task\" (string) and "
    "\"done\" (boolean), describing 6 chores.",
    "Produce a JSON object with a key \"steps\" mapping to an array of 6 strings, "
    "each step numbered in order.",
]

PROMPT_SETS = {
    "code": CODE_PROMPTS,
    "repetitive": REPETITIVE_PROMPTS,
    "structured": STRUCTURED_PROMPTS,
}


def _gpu_time(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _target_id_for(repo: str) -> int:
    table_path = Path("configs/target_ids.json")
    if table_path.exists():
        table = json.loads(table_path.read_text(encoding="utf-8"))
        if repo in table:
            return int(table[repo]["id"])
    return 0


# Fallback verification cost curve (matches runs/microbench/cost_curve.json).
VERIFY_CURVE = {1: 117.0, 2: 187.0, 4: 298.0, 8: 538.0, 16: 1041.0, 32: 2032.0, 64: 4040.0}


def _verify_curve() -> dict[int, float]:
    path = Path("runs/microbench/cost_curve.json")
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in record.get("verify_ms_by_nodes", {}).items()}
    return VERIFY_CURVE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--drafters", default="copy,general")
    parser.add_argument("--prompt-set", default="code", choices=sorted(PROMPT_SETS))
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument(
        "--dag", action="store_true", help="run the full router+multi-expert DAG system"
    )
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
    target_id = _target_id_for(args.repo)

    prompts = list(PROMPT_SETS[args.prompt_set])
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    drafters = {}
    copy_expert = CopyExpert(seed=4, min_length=3)
    if "copy" in args.drafters:
        drafters["copy"] = lambda: make_copy_drafter(copy_expert, args.spec_len)
    if "general" in args.drafters:
        trunk = build_trunk_from_config(Path("."))
        expert = GeneralExpert(trunk).to(device)
        if args.general_checkpoint:
            state = torch.load(args.general_checkpoint, map_location=device)
            expert.load_state_dict(state)
            print(f"loaded general checkpoint: {args.general_checkpoint}", flush=True)
        drafters["general"] = lambda: make_general_drafter(
            expert, args.spec_len, target_id, device
        )

    if args.dag:
        from familydraft.eval.draft_dag import (
            DagSpeculator,
            build_dag_router,
            make_macro_drafter,
            make_reject_memory_drafter,
        )
        from familydraft.experts.macro import MacroExpert
        from familydraft.experts.macro_render import build_renderer_from_config
        from familydraft.experts.reject_memory import RejectionMemory

        tok = target.tokenizer
        renderer = build_renderer_from_config(Path("."), tok, tok.vocab_size)
        macro_expert = MacroExpert(renderer, head=None)
        memory = RejectionMemory(min_support=1)
        dag_experts: dict = {}
        dag_horizons: dict = {}
        if "copy" in args.drafters:
            dag_experts["copy"] = make_copy_drafter(copy_expert, args.spec_len)
            dag_horizons["copy"] = args.spec_len
        if "general" in args.drafters:
            dag_experts["general"] = make_general_drafter(
                expert, args.spec_len, target_id, device
            )
            dag_horizons["general"] = args.spec_len
        dag_experts["macro"] = make_macro_drafter(macro_expert, tok)
        dag_horizons["macro"] = args.spec_len
        dag_experts["reject_memory"] = make_reject_memory_drafter(memory, target_id)
        dag_horizons["reject_memory"] = args.spec_len

        router_kwargs = dict(
            draft_ms={e: (10.0 if e == "general" else 1.0) for e in dag_experts},
            verify_curve=_verify_curve(),
            base={e: (3.0 if e == "copy" else 2.0) for e in dag_experts},
        )

        def _make_dag_spec():
            # Build a fresh speculator per prompt so router feedback does not
            # accumulate across prompts (mirrors the chain's fresh-per-prompt
            # measurement).
            router = build_dag_router(
                list(dag_experts.keys()),
                router_kwargs["draft_ms"],
                router_kwargs["verify_curve"],
                router_kwargs["base"],
                tau_abstain=0.01,
            )
            return DagSpeculator(
                target,
                router,
                {
                    "copy": make_copy_drafter(CopyExpert(seed=4, min_length=3), args.spec_len),
                    "macro": make_macro_drafter(macro_expert, tok),
                    "reject_memory": make_reject_memory_drafter(memory, target_id),
                    **(
                        {"general": drafters["general"]()}
                        if "general" in args.drafters
                        else {}
                    ),
                },
                {e: args.spec_len for e in dag_experts},
                target_id,
                memory=memory,
            )

        drafters["dag"] = _make_dag_spec
        print("dag system: router over", list(dag_experts.keys()), flush=True)

    warm_prompt = target.tokenizer("def f():", return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    target.generate_greedy(warm_prompt, 8)

    results = {
        "schema": "familydraft.integrated_speedup.v1",
        "repo": args.repo,
        "prompt_set": args.prompt_set,
        "max_new_tokens": args.max_new,
        "spec_len": args.spec_len,
        "general_checkpoint": args.general_checkpoint or None,
        "runs": [],
    }

    for prompt_text in prompts:
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
            made = factory()
            if hasattr(made, "generate"):
                speculator = made
            else:
                speculator = IntegratedSpeculator(target, made, args.spec_len, target_id)
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
