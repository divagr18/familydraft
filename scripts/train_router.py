"""Router rollout training, v1 (plan todo 20 - utility regression).

Runs the heterogeneous experts over training prompts and records, per decoding
step, the context features and the accepted-token length each expert's proposal
would have earned against the target. Fits per-expert ridge regression so the
router's expected-acceptance estimate reflects MEASURED quality instead of
cold-start guesses.

Usage (0.6B local):
  python scripts/train_router.py --repo Qwen/Qwen3-0.6B \
      --general-checkpoint runs/trainlogs/local_distill_clean/general_expert.pt \
      --prompts runs/traces_train_clean/worker_0.jsonl --out configs/router_weights.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

HORIZON = 4
FEATURE_DIM = 4
NUM_TARGETS = 7
RIDGE_LAMBDA = 1.0


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _features_from(text: str, copy_score: float, target_id: int) -> list[float]:
    from familydraft.experts.parse_state import parse_scan
    from familydraft.router.router import make_features

    state = parse_scan(text)
    parser_score = min(len(state.candidates) / 4.0, 1.0)
    return make_features(0.0, parser_score, copy_score, copy_score, target_id, NUM_TARGETS)


def _expert_proposal(expert_draft_fn, context_ids: list[int], horizon: int) -> list[int]:
    try:
        prop = list(expert_draft_fn(context_ids))[:horizon]
    except Exception:
        prop = []
    return prop


def _accepted_len(tgt, past, t_next, proposal: list[int], ctx_len: int) -> int:
    if not proposal:
        return 0
    cache_copy = copy.deepcopy(past)
    with torch.inference_mode():
        out = tgt.model(
            input_ids=torch.tensor([proposal], device=tgt.device),
            past_key_values=cache_copy,
            use_cache=True,
        )
    lg = out.logits[0]
    if proposal[0] != t_next:
        return 0
    m = 1
    for i in range(1, len(proposal)):
        if proposal[i] == int(torch.argmax(lg[i - 1], dim=-1)):
            m += 1
        else:
            break
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--prompts", default="runs/traces_train_clean/worker_0.jsonl")
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--steps-per-prompt", type=int, default=40)
    parser.add_argument("--out", default="configs/router_weights.json")
    args = parser.parse_args()

    from familydraft.draft.trunk import build_trunk_from_config
    from familydraft.eval.draft_dag import (
        make_macro_drafter,
        make_reject_memory_drafter,
    )
    from familydraft.eval.draft_loop import make_copy_drafter, make_general_drafter
    from familydraft.experts.copy import CopyExpert
    from familydraft.experts.general import GeneralExpert
    from familydraft.experts.macro import MacroExpert
    from familydraft.experts.macro_render import build_renderer_from_config
    from familydraft.experts.reject_memory import RejectionMemory
    from familydraft.targets.wrapper import TargetModel

    if not torch.cuda.is_available():
        print("train_router: CUDA GPU unavailable.", flush=True)
        return 2

    device = torch.device("cuda")
    tgt = TargetModel.load(args.repo, dtype="bf16")
    tok = tgt.tokenizer
    target_id = 0

    renderer = build_renderer_from_config(Path("."), tok, tok.vocab_size)
    macro_expert = MacroExpert(renderer, head=None)
    memory = RejectionMemory(min_support=1)
    copy_expert = CopyExpert(seed=4, min_length=3)

    experts: dict[str, object] = {
        "copy": make_copy_drafter(copy_expert, HORIZON),
        "macro": make_macro_drafter(macro_expert, tok),
        "reject_memory": make_reject_memory_drafter(memory, target_id),
    }
    if args.general_checkpoint:
        trunk = build_trunk_from_config(Path("."))
        gen = GeneralExpert(trunk).to(device)
        gen.load_state_dict(torch.load(args.general_checkpoint, map_location=device))
        experts["general"] = make_general_drafter(gen, HORIZON, target_id, device)

    # Load training prompts: --prompts may be a traces jsonl (rows with
    # prompt_ids), a plain {"prompt_text": ...} jsonl, or a directory of either.
    prompts_path = Path(args.prompts)
    files = sorted(prompts_path.glob("*.jsonl")) if prompts_path.is_dir() else [prompts_path]
    rows: list[dict] = []
    for f in files:
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    prompt_lists: list[list[int]] = []
    for r in rows[: args.n_prompts]:
        if "prompt_ids" in r:
            prompt_lists.append(r["prompt_ids"])
        elif "prompt_text" in r:
            prompt_lists.append(
                tok(r["prompt_text"], return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ][0].tolist()
            )
    if not prompt_lists:
        print(
            f"train_router: no prompts found under {prompts_path}. The traces "
            "live on the RunPod network volume (runs/traces_train/ is gitignored), "
            "or run trace generation first. Aborting.",
            flush=True,
        )
        return 3
    print(f"training prompts: {len(prompt_lists)}", flush=True)

    X: list[list[float]] = []
    Y: dict[str, list[float]] = {e: [] for e in experts}
    total_steps = 0

    for pids in prompt_lists:
        x0 = torch.tensor([pids], device=device)
        with torch.inference_mode():
            out = tgt.model(input_ids=x0, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        generated: list[int] = []
        ctx_len = len(pids)

        for _ in range(args.steps_per_prompt):
            t_next = int(torch.argmax(logits, dim=-1))
            context_ids = pids + generated
            text = tok.decode(context_ids, skip_special_tokens=True)
            copy_prop = _expert_proposal(experts["copy"], context_ids, HORIZON)
            copy_score = min(len(copy_prop) / HORIZON, 1.0)
            feats = _features_from(text, copy_score, target_id)
            X.append(feats)

            for name, draft_fn in experts.items():
                prop = _expert_proposal(draft_fn, context_ids, HORIZON)
                m = _accepted_len(tgt, past, t_next, prop, ctx_len)
                Y[name].append(float(m))

            generated.append(t_next)
            with torch.inference_mode():
                o = tgt.model(
                    input_ids=torch.tensor([[t_next]], device=device),
                    past_key_values=past,
                    use_cache=True,
                )
            past = o.past_key_values
            logits = o.logits[0, -1]
            ctx_len += 1
            total_steps += 1

    print(f"collected {total_steps} steps x {len(experts)} experts", flush=True)

    Xt = torch.tensor(X, dtype=torch.float32)
    weights: dict[str, list[float]] = {}
    eye = RIDGE_LAMBDA * torch.eye(Xt.shape[1])
    for name in experts:
        y = torch.tensor(Y[name], dtype=torch.float32)
        A = Xt.t() @ Xt + eye
        b = Xt.t() @ y
        w = torch.linalg.solve(A, b)
        weights[name] = w.tolist()
        pred = (Xt @ w).mean().item()
        print(f"  {name}: mean measured acceptance={y.mean().item():.3f} "
              f"predicted={pred:.3f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(weights, indent=1), encoding="utf-8")
    print(f"wrote router weights -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
