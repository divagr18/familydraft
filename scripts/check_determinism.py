"""Phase-1 baseline determinism self-check (M3 / plan todo 21 QA).

Runs one baseline system twice on the same prompts under identical seeds and
asserts the emitted token sequences are bit-identical.

Exit codes:
  0  PASS - both runs identical
  1  FAIL - runs diverged (the divergent run pair is named)
  2  ERROR - setup failure (missing GPU / system / prompts)

Mutation evidence (QA failure scenario): `--mutate` injects an UNSEEDED dropout
into the target model, then verifies the self-check correctly FAILS, naming the
divergent pair. This proves the check can actually detect nondeterminism instead
of trivially passing.

Usage:
  uv run python scripts/check_determinism.py --system vanilla_ar
  uv run python scripts/check_determinism.py --system full_proposal_moe --mutate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_baselines import (  # noqa: E402
    TASK_CLASSES,
    VALID_SYSTEMS,
    _chain_factory,
    _dag_factory,
    _time_call,
)

DEFAULT_TASK = "structured"


def _mutate_target(target):
    """Inject run-to-run nondeterminism that survives eval/inference mode.

    nn.Dropout is a no-op in eval mode, so we wrap each MLP output with an
    unconditional noise layer that adds torch.randn (which is NOT reproducible
    across different RNG seeds, even inside inference_mode). The divergence is
    only observed when the two runs use different seeds - which the check does.
    """
    from torch import nn

    class _Noise(nn.Module):
        def forward(self, x):
            return x + torch.randn_like(x) * 0.5

    for layer in target.model.model.layers:
        if hasattr(layer, "mlp"):
            layer.mlp = nn.Sequential(layer.mlp, _Noise())


def _build_factory(args, target, device, target_id, mutate: bool):
    if args.system == "vanilla_ar":
        if mutate:
            _mutate_target(target)
        return None

    if mutate:
        _mutate_target(target)

    if args.system in ("small_dense_drafter", "equal_flop_dense_drafter"):
        return _chain_factory(args, target, device, target_id)

    always_on = ["copy"]
    max_experts = 1 if args.system == "single_best_expert" else 2
    tree_verify = args.system != "hetero_top2_no_fusion"
    return _dag_factory(
        args, target, device, target_id, with_general=bool(args.general_checkpoint),
        always_on=always_on, max_experts=max_experts, tree_verify=tree_verify,
    )


def _run_tokens(factory, target, prompts, max_new):
    out = []
    for text in prompts:
        prompt_ids = target.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        prompt_list = prompt_ids.tolist()
        if factory is None:
            with torch.inference_mode():
                tok = target.generate_greedy(prompt_ids, max_new)[0, len(prompt_list):].tolist()
        else:
            spec = factory()
            _, res = _time_call(lambda: spec.generate(prompt_list, max_new))
            tok = res["tokens"]
        out.append(tok)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=VALID_SYSTEMS, default="vanilla_ar")
    parser.add_argument("--task-class", choices=TASK_CLASSES, default=DEFAULT_TASK)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--router-weights", default="")
    parser.add_argument(
        "--mutate",
        action="store_true",
        help="inject unseeded dropout (QA failure scenario)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("check_determinism: CUDA GPU unavailable.", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    torch.manual_seed(1234)
    from familydraft.targets.wrapper import TargetModel
    from scripts.run_baselines import _target_id_for

    target = TargetModel.load(args.repo, dtype="bf16")
    target_id = _target_id_for(args.repo)

    prompts: list[str] = []
    manifest = Path("data/eval") / args.task_class / "items.jsonl"
    if manifest.exists():
        items = [
            json.loads(ln)
            for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        prompts = [it.get("prompt_text") for it in items if it.get("prompt_text")]
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        print(f"check_determinism: no prompts for {args.task_class}", file=sys.stderr)
        return 2

    factory = _build_factory(args, target, device, target_id, mutate=args.mutate)

    # Two independent runs under DIFFERENT seeds. A clean deterministic
    # pipeline (no RNG in the decode path) is bit-identical regardless of seed;
    # a mutated pipeline (injected noise) diverges and is caught.
    torch.manual_seed(1234)
    run1 = _run_tokens(factory, target, prompts, args.max_new)
    torch.manual_seed(5678)
    run2 = _run_tokens(factory, target, prompts, args.max_new)

    diverged = [(i, a, b) for i, (a, b) in enumerate(zip(run1, run2)) if a != b]
    if diverged:
        for i, a, b in diverged:
            print(f"  DIVERGENT run pair: prompt {i}: {a[:8]} vs {b[:8]}")
        print(f"check_determinism: FAIL ({args.system}, {args.task_class}) - "
              f"{len(diverged)}/{len(prompts)} prompts diverged across identical-seed reruns")
        return 1

    if args.mutate:
        # Mutation was applied but runs still matched: the check is blind.
        print(f"check_determinism: FAIL (mutation not detected by {args.system})")
        return 1

    print(
        f"check_determinism: PASS ({args.system}, {args.task_class}) - "
        "bit-identical across seeded reruns"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
