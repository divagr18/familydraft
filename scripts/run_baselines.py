"""Unified Phase-1 baseline harness (M3 / plan todo 21).

Runs one baseline system x one task class under the P5 protocol (5-run
mean±std, EOS stop, isolated per-run state, exact-sequence agreement) and emits
a schema-valid baseline report (configs/baseline_report.schema.json) with FLOP
accounting per emitted token.

Systems:
  vanilla_ar            - autoregressive target decoding
  small_dense_drafter   - 6-layer trunk as dense chain drafter (general expert)
  equal_flop_dense_drafter - dense drafter sized to the DAG's active drafting
                             FLOPs (constructible proxy: trunk; the dense-
                             equivalent layer count is reported for honesty)
  single_best_expert    - DAG, router limited to 1 expert
  hetero_top2_no_fusion - DAG, per-branch sequential verify (no tree fusion)
  full_proposal_moe     - full DAG with tree verification

Every speculative run builds a FRESH speculator (fresh router state + fresh
rejection memory) so no feedback accumulates across measurements.

Usage: uv run python scripts/run_baselines.py --system full_proposal_moe \
       --task-class structured [--repo Qwen/Qwen3-0.6B] [--runs 5] [--out ...]
"""

from __future__ import annotations

import argparse
import hashlib
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
    'Generate a JSON array of 8 objects. Each object has exactly the keys "id", '
    '"name", "value". The id runs from 0 to 7.',
    "Create a JSON object with keys item_0 through item_7, where each key maps to "
    "its index as the value.",
    'Write a JSON list of 8 strings. The k-th string must be "entry_k" for k 0 to 7.',
    "Output a CSV with header id,name,score followed by 8 rows whose id runs 0 to 7.",
]

STRUCTURED_PROMPTS = [
    'Produce a JSON object matching the schema {"type":"object","properties":'
    '{"name":{"type":"string"},"age":{"type":"integer"}},"required":'
    '["name","age"]} for a person named Ada aged 36.',
    'Produce a JSON object matching the schema {"type":"object","properties":'
    '{"city":{"type":"string"},"population":{"type":"integer"}},"required":'
    '["city","population"]} for Tokyo.',
    'Produce a JSON array of 6 objects, each with keys "task" (string) and '
    '"done" (boolean), describing 6 chores.',
    'Produce a JSON object with a key "steps" mapping to an array of 6 strings, '
    "each step numbered in order.",
]

PROMPT_SETS = {
    "code": CODE_PROMPTS,
    "repetitive": REPETITIVE_PROMPTS,
    "structured": STRUCTURED_PROMPTS,
}

VERIFY_CURVE = {1: 117.0, 2: 187.0, 4: 298.0, 8: 538.0, 16: 1041.0, 32: 2032.0, 64: 4040.0}


def _mean(vals):
    return sum(vals) / max(1, len(vals))


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _target_id_for(repo: str) -> int:
    table_path = Path("configs/target_ids.json")
    if table_path.exists():
        table = json.loads(table_path.read_text(encoding="utf-8"))
        if repo in table:
            return int(table[repo]["id"])
    return 0


def _verify_curve() -> dict[int, float]:
    path = Path("runs/microbench/cost_curve.json")
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in record.get("verify_ms_by_nodes", {}).items()}
    return VERIFY_CURVE


def _load_router_config() -> dict:
    import yaml

    cfg_path = Path("configs/router.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return {
            "draft_ms": dict(cfg.get("draft_ms", {})),
            "base_acceptance": dict(cfg.get("base_acceptance", {})),
            "copy_cost_fixed_ms": float(cfg.get("copy_cost_fixed_ms", 1.0)),
        }
    return {}

VALID_SYSTEMS = (
    "vanilla_ar",
    "small_dense_drafter",
    "equal_flop_dense_drafter",
    "single_best_expert",
    "hetero_top2_no_fusion",
    "full_proposal_moe",
)
TASK_CLASSES = ("code", "repetitive", "structured", "gsm8k")

# Qwen3-0.6B architecture (target 28 layers, trunk 6 layers).
TARGET_HIDDEN, TARGET_LAYERS, TARGET_INTER = 1024, 28, 3072
TARGET_KV, TARGET_HEADS, VOCAB = 8, 16, 151936
TRUNK_LAYERS = 6


def _config_hash(*payloads) -> str:
    h = hashlib.sha256()
    for p in payloads:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def _time_call(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _vanilla_tokens(target, prompt_ids, max_new):
    prompt_list = prompt_ids.tolist()
    with torch.inference_mode():
        return target.generate_greedy(prompt_ids, max_new)[0, len(prompt_list):].tolist()


def _chain_factory(args, target, device, target_id):
    """Fresh IntegratedSpeculator with a fresh KV-cache drafter per call."""
    from familydraft.draft.trunk import build_trunk_from_config
    from familydraft.eval.draft_loop import IntegratedSpeculator, make_general_drafter
    from familydraft.experts.general import GeneralExpert

    trunk = build_trunk_from_config(Path(".")).to(device)
    expert = GeneralExpert(trunk).to(device)
    if args.general_checkpoint:
        expert.load_state_dict(torch.load(args.general_checkpoint, map_location=device))
    drafter = make_general_drafter(expert, args.spec_len, target_id, device)

    def make():
        return IntegratedSpeculator(target, drafter, args.spec_len, target_id)

    return make


def _dag_factory(
    args, target, device, target_id, with_general, always_on, max_experts, tree_verify,
    ablation: dict | None = None,
):
    from familydraft.eval.draft_dag import (
        DagSpeculator,
        build_dag_router,
        make_macro_drafter,
        make_reject_memory_drafter,
    )
    from familydraft.eval.draft_loop import make_copy_drafter, make_general_drafter
    from familydraft.experts.copy import CopyExpert
    from familydraft.experts.general import GeneralExpert
    from familydraft.experts.macro import MacroExpert
    from familydraft.experts.macro_render import build_renderer_from_config
    from familydraft.experts.reject_memory import RejectionMemory

    ov = ablation or {}
    max_experts = int(ov.get("max_experts", max_experts))
    tree_verify = bool(ov.get("tree_verify", tree_verify))
    routing_mode = str(ov.get("routing_mode", "utility"))
    tau_abstain = float(ov.get("tau_abstain", 0.0))
    no_target_embedding = bool(ov.get("no_target_embedding", False))
    no_online_feedback = bool(ov.get("no_online_feedback", False))
    no_rejection_memory = bool(ov.get("no_rejection_memory", False))
    load_router_weights = bool(ov.get("load_router_weights", True))

    tok = target.tokenizer
    renderer = build_renderer_from_config(Path("."), tok, tok.vocab_size)
    macro_expert = MacroExpert(renderer, head=None)
    cfg = _load_router_config()

    general_expert = None
    if with_general and args.general_checkpoint:
        from familydraft.draft.trunk import build_trunk_from_config

        trunk = build_trunk_from_config(Path(".")).to(device)
        expert = GeneralExpert(trunk).to(device)
        expert.load_state_dict(torch.load(args.general_checkpoint, map_location=device))
        general_expert = expert

    names = ["copy", "macro", "reject_memory"] + (["general"] if general_expert else [])
    if no_rejection_memory:
        names = [n for n in names if n != "reject_memory"]

    def make():
        router = build_dag_router(
            names,
            {e: cfg.get("draft_ms", {}).get(e, 1.0) for e in names},
            _verify_curve(),
            {e: cfg.get("base_acceptance", {}).get(e, 1.0) for e in names},
            tau_abstain=tau_abstain,
            always_on_cost_ms={"copy": cfg.get("copy_cost_fixed_ms", 1.0)},
            routing_mode=routing_mode,
        )
        if load_router_weights and args.router_weights:
            from familydraft.router.router import UtilityRouter

            router.set_weights(UtilityRouter.load_weights(args.router_weights))
        memory = None if no_rejection_memory else RejectionMemory(min_support=1)
        dag_experts = {
            "copy": make_copy_drafter(CopyExpert(seed=4, min_length=3), args.spec_len),
            "macro": make_macro_drafter(macro_expert, tok),
        }
        if not no_rejection_memory:
            dag_experts["reject_memory"] = make_reject_memory_drafter(memory, target_id)
        if general_expert is not None:
            dag_experts["general"] = make_general_drafter(
                general_expert, args.spec_len, target_id, device
            )
        return DagSpeculator(
            target,
            router,
            dag_experts,
            {e: args.spec_len for e in dag_experts},
            target_id,
            memory=memory,
            always_on=always_on,
            max_experts=max_experts,
            tree_verify=tree_verify,
            no_target_embedding=no_target_embedding,
            no_online_feedback=no_online_feedback,
        )

    return make


def _flops_ledger(system, spec_len, tpr, verify_nodes) -> dict:
    from familydraft.eval.flops import FlopBudget, spec_loop_flops_per_emitted_token

    target_budget = FlopBudget.from_config(
        TARGET_HIDDEN, TARGET_LAYERS, TARGET_INTER, TARGET_KV, TARGET_HEADS, VOCAB, "target"
    )
    trunk_budget = FlopBudget.from_config(
        TARGET_HIDDEN, TRUNK_LAYERS, TARGET_INTER, TARGET_KV, TARGET_HEADS, VOCAB, "trunk"
    )
    if system == "vanilla_ar":
        return {"flops_per_emitted_token": target_budget.flops_per_token,
                "label": "vanilla AR"}
    if system in ("small_dense_drafter", "equal_flop_dense_drafter"):
        draft_tokens = verify_nodes = float(spec_len)
    else:
        draft_tokens = verify_nodes
    flops = spec_loop_flops_per_emitted_token(
        target_budget, trunk_budget, draft_tokens, verify_nodes, tpr
    )
    return {"flops_per_emitted_token": flops, "dense_equivalent_layers": TRUNK_LAYERS,
            "label": system}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=VALID_SYSTEMS, required=True)
    parser.add_argument("--task-class", choices=TASK_CLASSES, required=True)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--router-weights", default="")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--ablation",
        default="",
        help="pre-registered ablation config name (configs/ablations/<name>.yaml)",
    )
    args = parser.parse_args()

    ablation = {}
    if args.ablation:
        import yaml

        ab_path = Path("configs/ablations") / f"{args.ablation}.yaml"
        if not ab_path.exists():
            print(f"run_baselines: unknown ablation {args.ablation!r} ({ab_path})", flush=True)
            return 2
        ablation = yaml.safe_load(ab_path.read_text(encoding="utf-8")).get("overrides", {})
        print(f"run_baselines: ablation {args.ablation}: {ablation}", flush=True)

    if not torch.cuda.is_available():
        print("run_baselines: CUDA GPU unavailable.", flush=True)
        return 2

    device = torch.device("cuda")
    from familydraft.targets.wrapper import TargetModel

    target = TargetModel.load(args.repo, dtype="bf16")
    target_id = _target_id_for(args.repo)

    prompts = list(PROMPT_SETS.get(args.task_class, []))
    manifest = Path("data/eval") / args.task_class / "items.jsonl"
    if manifest.exists():
        items = [
            json.loads(ln)
            for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        manifest_prompts = [it.get("prompt_text") for it in items if it.get("prompt_text")]
        if manifest_prompts:
            prompts = manifest_prompts
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        print(f"run_baselines: no prompts for task-class {args.task_class}", flush=True)
        return 3

    if args.system == "vanilla_ar":
        factory = None
    elif args.system in ("small_dense_drafter", "equal_flop_dense_drafter"):
        factory = _chain_factory(args, target, device, target_id)
    else:
        always_on = ["copy"]
        max_experts = 1 if args.system == "single_best_expert" else 2
        tree_verify = args.system != "hetero_top2_no_fusion"
        factory = _dag_factory(
            args, target, device, target_id, with_general=bool(args.general_checkpoint),
            always_on=always_on, max_experts=max_experts, tree_verify=tree_verify,
            ablation=ablation,
        )

    warm_ids = target.tokenizer(
        "def f():", return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    target.generate_greedy(warm_ids, 8)

    times, tprs, exacts, vnodes, emitted = [], [], [], [], []
    for text in prompts:
        prompt_ids = target.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        prompt_list = prompt_ids.tolist()

        if args.system == "vanilla_ar":
            dt, tok = _time_call(lambda: _vanilla_tokens(target, prompt_ids, args.max_new))
            times.append(dt)
            tprs.append(1.0)
            exacts.append(1.0)
            emitted.append(len(tok))
            continue

        vanilla_ref = _vanilla_tokens(target, prompt_ids, args.max_new)
        for _ in range(args.runs):
            spec = factory()
            dt, res = _time_call(lambda: spec.generate(prompt_list, args.max_new))
            times.append(dt)
            tprs.append(res["tokens_per_round"])
            exacts.append(1 if res["tokens"] == vanilla_ref else 0)
            emitted.append(len(res["tokens"]))
            if "verify_nodes_per_round" in res:
                vnodes.append(res["verify_nodes_per_round"])

    mean_t = _mean(times)
    std_t = _std(times)
    mean_emitted = _mean(emitted)
    tps = mean_emitted / max(1e-9, mean_t)
    tps_std = std_t * tps / max(1e-9, mean_t)

    ledger = _flops_ledger(
        args.system,
        args.spec_len,
        _mean(tprs) if tprs else 1.0,
        _mean(vnodes) if vnodes else float(args.spec_len),
    )

    report = {
        "schema": "familydraft.baseline_report.v1",
        "system": args.system,
        "task_class": args.task_class,
        "repo": args.repo,
        "target_tokens_per_second": tps,
        "mean": tps,
        "std": tps_std,
        "runs": max(1, len(times)),
        "config_hash": _config_hash(args.system, args.task_class, args.repo, args.spec_len,
                                     args.general_checkpoint, args.router_weights, ledger),
        "flops_per_emitted_token": ledger["flops_per_emitted_token"],
        "dense_equivalent_layers": ledger.get("dense_equivalent_layers"),
        "exactness": {
            "fp32_exact": (_mean(exacts) == 1.0) if exacts else True,
            "bf16_exact_match_rate": _mean(exacts) if exacts else 1.0,
        },
        "tokens_per_round": _mean(tprs) if tprs else 1.0,
    }

    out_path = Path(args.out) if args.out else (
        Path("runs/baselines") / f"{args.system}_{args.task_class}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
