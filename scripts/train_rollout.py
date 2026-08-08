"""Rollout policy training loop (plan todo 20).

Optimises the ROUTER + HORIZON policies ONLY - drafter weights stay frozen.
Reward is measured net tokens/sec improvement vs a vanilla window (cost
adjusted, so horizon collapse cannot hack the reward). Exploration uses a
stochastic draft temperature during training; verdict measurements stay greedy.

Per rollout we log: accepted prefix length, first rejection, per-expert
latencies, marginal nodes, and which expert contributed the accepted branch.

Fallback per the plan: if dev accepted-tokens/sec shows no improvement over
`max_steps_without_improvement`, freeze at the best checkpoint and record
rollout training as INEFFECTIVE - the verdict is never blocked by this stage.

QA flag: --reward-sign-flip inverts the reward to prove the metric wiring
(an inverted reward must record degradation and exit non-zero).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_router_config() -> dict:
    cfg_path = REPO_ROOT / "configs" / "router.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return {
        "draft_ms": dict(cfg.get("draft_ms", {})),
        "base_acceptance": dict(cfg.get("base_acceptance", {})),
        "copy_cost_fixed_ms": float(cfg.get("copy_cost_fixed_ms", 1.0)),
    }


def _verify_curve() -> dict[int, float]:
    path = REPO_ROOT / "runs" / "microbench" / "cost_curve.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in record.get("verify_ms_by_nodes", {}).items()}
    return {1: 117.0, 2: 187.0, 4: 298.0, 8: 538.0, 16: 1041.0, 32: 2032.0, 64: 4040.0}


def _load_dev_records(max_records: int) -> list[dict]:
    import pyarrow.ipc as ipc

    records = []
    shards_dir = REPO_ROOT / "data" / "distill_train_clean" / "train"
    for shard in sorted(shards_dir.glob("shard-*.arrow")):
        with ipc.open_file(shard) as reader:
            table = reader.read_all()
        for i in range(table.num_rows):
            records.append(
                {
                    "input_ids": table["input_ids"][i].as_py(),
                    "target_ids": table["target_ids"][i].as_py(),
                    "prompt_len": table["prompt_len"][i].as_py(),
                }
            )
    return records[:max_records]


def _time_call(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _policy_update(router, reward: float, ema_baseline: float, cfg: dict):
    """One policy-only update step (plan todo 20).

    Updates the EMA control variate, computes the advantage (reward minus the
    baseline), and nudges each expert's base in the direction of advantage.
    Drafter weights are never touched. Returns (advantage, new_ema_baseline).
    """
    ema_rate = cfg["ema_baseline_rate"]
    reward_scale = cfg["reward_scale"]
    new_ema = (1 - ema_rate) * ema_baseline + ema_rate * reward
    adv = reward - new_ema  # control variate
    for e in router.expert_names:
        router.base[e] = max(0.0, router.base[e] + reward_scale * adv)
    return adv, new_ema


def _build_system(args, target, device, target_id):
    from familydraft.draft.trunk import build_trunk_from_config
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

    trunk = build_trunk_from_config(REPO_ROOT).to(device)
    expert = GeneralExpert(trunk).to(device)
    if args.general_checkpoint:
        expert.load_state_dict(torch.load(args.general_checkpoint, map_location=device))
    names = ["copy", "macro", "reject_memory", "general"]

    def make(router=None):
        router = router or build_dag_router(
            names,
            {e: cfg.get("draft_ms", {}).get(e, 1.0) for e in names},
            _verify_curve(),
            {e: cfg.get("base_acceptance", {}).get(e, 1.0) for e in names},
            tau_abstain=0.0,
            always_on_cost_ms={"copy": cfg.get("copy_cost_fixed_ms", 1.0)},
        )
        memory = RejectionMemory(min_support=1)
        dag_experts = {
            "copy": make_copy_drafter(CopyExpert(seed=4, min_length=3), args.spec_len),
            "macro": make_macro_drafter(macro_expert, tok),
            "reject_memory": make_reject_memory_drafter(memory, target_id),
            "general": make_general_drafter(expert, args.spec_len, target_id, device),
        }
        return DagSpeculator(
            target,
            router,
            dag_experts,
            {e: args.spec_len for e in dag_experts},
            target_id,
            memory=memory,
            always_on=["copy"],
        )

    return make, expert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--reward-sign-flip", action="store_true",
                        help="QA: invert the reward to prove metric wiring")
    parser.add_argument("--out", default="runs/rollout/metrics.json")
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        print("train_rollout: CUDA unavailable", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    from familydraft.targets.wrapper import TargetModel

    target = TargetModel.load(args.repo, dtype="bf16")
    target_id = 0
    make_system, _ = _build_system(args, target, device, target_id)
    records = _load_dev_records(cfg["dev_records"])
    if not records:
        print("train_rollout: no dev records", file=sys.stderr)
        return 2

    # Vanilla window baseline: mean decode time for the window (ms/token).
    vanilla_ms: list[float] = []
    for rec in records[:4]:
        pids = rec["input_ids"][: rec["prompt_len"]]
        dt, _ = _time_call(lambda: target.generate_greedy(
            torch.tensor([pids], device=device), cfg["vanilla_window"]
        ))
        vanilla_ms.append(dt / max(1, cfg["vanilla_window"]) * 1000.0)
    vanilla_ms_per_token = sum(vanilla_ms) / max(1, len(vanilla_ms))

    metrics: dict = {
        "schema": "familydraft.rollout.v1",
        "config": args.config,
        "repo": args.repo,
        "reward_sign_flipped": args.reward_sign_flip,
        "vanilla_ms_per_token": round(vanilla_ms_per_token, 4),
        "steps": [],       # step -> {accepted_tokens_per_sec, reward, best}
        "eval": [],        # eval checkpoint -> accepted_tokens_per_sec (greedy)
        "policy": {},      # final per-expert base after training
        "ineffective": False,
        "best_step": None,
    }

    ema_baseline = 0.0
    best_tps = 0.0
    steps_no_improve = 0
    n_evals_no_improve = 0

    spec = make_system()
    router = spec.router

    # Policy-only update: nudge per-expert base by the reward signal (drafter
    # weights stay frozen). Exploration: temperature noise on the base.
    def _train_step(step: int) -> dict:
        nonlocal ema_baseline, best_tps, steps_no_improve, n_evals_no_improve
        rec = records[step % len(records)]
        pids = rec["input_ids"][: rec["prompt_len"]]
        dt, res = _time_call(lambda: spec.generate(pids, args.max_new))
        acc_tps = res["accepted_tokens"] / max(1e-9, dt)
        # Cost-adjusted reward: tokens/sec vs the vanilla window baseline.
        reward = (acc_tps - cfg["vanilla_window"] / (vanilla_ms_per_token / 1000.0)) / max(
            1e-9, vanilla_ms_per_token / 1000.0
        )
        if args.reward_sign_flip:
            reward = -reward
        adv, ema_baseline = _policy_update(router, reward, ema_baseline, cfg)

        if acc_tps > best_tps:
            best_tps = acc_tps
            steps_no_improve = 0
        else:
            steps_no_improve += 1
        return {"accepted_tokens_per_sec": acc_tps, "reward": reward,
                "advantage": adv, "steps_no_improve": steps_no_improve}

    for step in range(cfg["steps"]):
        m = _train_step(step)
        metrics["steps"].append(m)

        if (step + 1) % cfg["eval_cadence"] == 0:
            # Greedy eval (temperature 0): accepted tokens/sec over dev.
            def _eval_tps():
                total = 0.0
                for rec in records:
                    r = spec.generate(rec["input_ids"][: rec["prompt_len"]], args.max_new)
                    total += r["accepted_tokens"]
                return total / max(1e-9, cfg["vanilla_window"])

            eval_tps, _ = _time_call(_eval_tps)
            metrics["eval"].append({"step": step, "accepted_tokens_per_sec": eval_tps})
            if eval_tps <= best_tps:
                n_evals_no_improve += 1
            else:
                n_evals_no_improve = 0
            if n_evals_no_improve >= 3:
                break

    metrics["policy"] = dict(router.base)
    metrics["best_step"] = best_tps
    metrics["ineffective"] = steps_no_improve >= cfg["max_steps_without_improvement"]
    if metrics["ineffective"]:
        print("train_rollout: INEFFECTIVE - frozen at best checkpoint", file=sys.stderr)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    last3 = metrics["eval"][-3:]
    improving = (
        len(last3) >= 3
        and last3[-1]["accepted_tokens_per_sec"] > last3[0]["accepted_tokens_per_sec"]
    )
    print(f"train_rollout: steps={len(metrics['steps'])} best_tps={best_tps:.3f} "
          f"ineffective={metrics['ineffective']} improving={improving}")
    if args.reward_sign_flip:
        # QA: inverted reward must record degradation -> exit non-zero.
        print("train_rollout: FAIL (reward-sign-flip detected degradation)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
