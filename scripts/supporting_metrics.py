"""Collect per-expert supporting metrics (plan todo 23).

Runs the DAG speculator directly and aggregates the per-expert instrumentation
(winner acceptance, proposals made, abstentions, second-expert wins) across
prompts and runs, then computes the plan's supporting tables:

  - per-expert proposal utility = winner_accepted / (draft_ms + verify_ms*horizon)
  - second-expert marginal wins (rounds where a non-always-on expert's branch
    won while >=2 experts were selected)
  - abstention rate and (where derivable) precision/recall

Writes runs/results/supporting_metrics.json and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baselines import (  # noqa: E402
    PROMPT_SETS,
    TASK_CLASSES,
    _load_router_config,
    _target_id_for,
    _verify_curve,
)


def _dag_factory(args, target, device, target_id):
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

    tok = target.tokenizer
    renderer = build_renderer_from_config(REPO_ROOT, tok, tok.vocab_size)
    macro_expert = MacroExpert(renderer, head=None)
    cfg = _load_router_config()

    general_expert = None
    if args.general_checkpoint:
        from familydraft.draft.trunk import build_trunk_from_config

        trunk = build_trunk_from_config(REPO_ROOT).to(device)
        expert = GeneralExpert(trunk).to(device)
        expert.load_state_dict(torch.load(args.general_checkpoint, map_location=device))
        general_expert = expert

    names = ["copy", "macro", "reject_memory"] + (["general"] if general_expert else [])

    def make():
        router = build_dag_router(
            names,
            {e: cfg.get("draft_ms", {}).get(e, 1.0) for e in names},
            _verify_curve(),
            {e: cfg.get("base_acceptance", {}).get(e, 1.0) for e in names},
            tau_abstain=0.0,
            always_on_cost_ms={"copy": cfg.get("copy_cost_fixed_ms", 1.0)},
        )
        if args.router_weights:
            from familydraft.router.router import UtilityRouter

            router.set_weights(UtilityRouter.load_weights(args.router_weights))
        memory = RejectionMemory(min_support=1)
        dag_experts = {
            "copy": make_copy_drafter(CopyExpert(seed=4, min_length=3), args.spec_len),
            "macro": make_macro_drafter(macro_expert, tok),
            "reject_memory": make_reject_memory_drafter(memory, target_id),
        }
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
            always_on=["copy"],
        )

    return make


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-class", choices=TASK_CLASSES, default="structured")
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--router-weights", default="")
    parser.add_argument("--out", default="runs/results/supporting_metrics.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("supporting_metrics: CUDA unavailable", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    from familydraft.targets.wrapper import TargetModel

    target = TargetModel.load(args.repo, dtype="bf16")
    target_id = _target_id_for(args.repo)
    factory = _dag_factory(args, target, device, target_id)

    prompts = list(PROMPT_SETS.get(args.task_class, []))
    manifest = REPO_ROOT / "data" / "eval" / args.task_class / "items.jsonl"
    if manifest.exists():
        items = [
            json.loads(ln)
            for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        mp = [it.get("prompt_text") for it in items if it.get("prompt_text")]
        if mp:
            prompts = mp
    prompts = prompts[: args.max_prompts]

    agg: dict = {
        "rounds": 0, "abstain_events": 0, "selected_events": 0, "second_rank_wins": 0,
        "accepted_tokens": 0, "winner_acceptance": {}, "proposals_made": {},
    }
    for _ in range(args.runs):
        for text in prompts:
            pids = target.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].tolist()
            spec = factory()
            res = spec.generate(pids, args.max_new)
            agg["rounds"] += res["rounds"]
            agg["abstain_events"] += res["abstain_events"]
            agg["selected_events"] += res["selected_events"]
            agg["second_rank_wins"] += res["second_rank_wins"]
            agg["accepted_tokens"] += res["accepted_tokens"]
            for e, v in res.get("winner_acceptance", {}).items():
                agg["winner_acceptance"][e] = agg["winner_acceptance"].get(e, 0) + v
            for e, v in res.get("proposals_made", {}).items():
                agg["proposals_made"][e] = agg["proposals_made"].get(e, 0) + v

    cfg = _load_router_config()
    draft_ms = cfg.get("draft_ms", {})
    verify_curve = _verify_curve()
    nodes = sorted(int(k) for k in verify_curve)
    if len(nodes) >= 2:
        verify_ms = (verify_curve[nodes[-1]] - verify_curve[nodes[0]]) / max(
            1, nodes[-1] - nodes[0]
        )
    else:
        verify_ms = 1.0

    rows = []
    for e in sorted(agg["winner_acceptance"]):
        accepted = agg["winner_acceptance"][e]
        proposals = agg["proposals_made"].get(e, 0)
        cost = draft_ms.get(e, 1.0) + verify_ms * args.spec_len
        utility = accepted / max(1e-9, cost)
        rows.append({
            "expert": e, "winner_accepted_tokens": accepted,
            "proposals_made": proposals,
            "draft_plus_verify_ms": round(cost, 3),
            "proposal_utility": round(utility, 4),
        })

    out = {
        "schema": "familydraft.supporting_metrics.v1",
        "task_class": args.task_class,
        "repo": args.repo,
        "runs": args.runs,
        "prompts_per_run": len(prompts),
        "aggregate": agg,
        "verify_ms_per_node": verify_ms,
        "proposal_utility_table": rows,
        "abstention_rate": round(agg["abstain_events"] / max(1, agg["rounds"]), 4),
        "second_rank_win_rate": round(
            agg["second_rank_wins"] / max(1, agg["selected_events"]), 4
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"rounds={agg['rounds']} abstain={agg['abstain_events']} "
          f"selected={agg['selected_events']} second_rank_wins={agg['second_rank_wins']}")
    print(
        f"abstention_rate={out['abstention_rate']} "
        f"second_rank_win_rate={out['second_rank_win_rate']}"
    )
    for r in rows:
        print(f"  {r['expert']}: utility={r['proposal_utility']} "
              f"(accepted={r['winner_accepted_tokens']} proposals={r['proposals_made']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
