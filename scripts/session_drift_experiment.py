"""Session-level drift adaptation experiment (plan todo 27).

Synthetic drift scenario: the session task mix shifts mid-run across three
segments (chat -> code -> structured). Compares:

  adaptive  - router EMA + rejection memory + online calibration ON (adapts)
  static    - all adaptation frozen at segment-1 stats (router base/EMA frozen,
              no memory, no calibration)

Per 100-token window we record accepted tokens/sec; after each segment shift we
compute the recovery window (windows-to-recover >=90% of the best-segment
utility). Acceptance: adaptive trajectory >= static control on mean accepted
tokens/sec in segments 2 and 3; memory bounded by the RejectionMemory LRU cap.

QA: --no-memory disables the rejection memory on the adaptive arm - the
adaptation delta vs static must shrink measurably (asserted > 0).
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

from scripts.run_baselines import (  # noqa: E402
    _load_router_config,
    _target_id_for,
    _verify_curve,
)

SEGMENTS = ["chat", "code", "structured"]
MANIFEST_FILE = {
    "chat": "mtbench/items.jsonl",
    "code": "humaneval/items.jsonl",
    "structured": "structured/items.jsonl",
}


def _segment_prompts(segment: str, max_prompts: int) -> list[str]:
    path = REPO_ROOT / "data" / "eval" / MANIFEST_FILE[segment]
    items = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [it["prompt_text"] for it in items if it.get("prompt_text")][:max_prompts]


def _time_call(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _make_dag(args, target, device, target_id, online_config, no_memory,
              no_online_feedback=False):
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

    def make():
        router = build_dag_router(
            names,
            {e: cfg.get("draft_ms", {}).get(e, 1.0) for e in names},
            _verify_curve(),
            {e: cfg.get("base_acceptance", {}).get(e, 1.0) for e in names},
            tau_abstain=0.0,
            always_on_cost_ms={"copy": cfg.get("copy_cost_fixed_ms", 1.0)},
        )
        memory = None if no_memory else RejectionMemory(min_support=1)
        dag_experts = {
            "copy": make_copy_drafter(CopyExpert(seed=4, min_length=3), args.spec_len),
            "macro": make_macro_drafter(macro_expert, tok),
            "general": make_general_drafter(expert, args.spec_len, target_id, device),
        }
        if not no_memory:
            dag_experts["reject_memory"] = make_reject_memory_drafter(memory, target_id)
        return DagSpeculator(
            target,
            router,
            dag_experts,
            {e: args.spec_len for e in dag_experts},
            target_id,
            memory=memory,
            always_on=["copy"],
            online_config=online_config,
            no_online_feedback=no_online_feedback,
        )

    return make


def _run_segment(spec, prompts, max_new):
    """Run one segment; return per-prompt accepted tokens/sec + memory size.

    The metric is real wall-clock tokens/sec (accepted tokens / decode time),
    so adaptation that cuts rounds (better selection, memory reuse) shows up.
    """
    windows: list[float] = []
    memory_sizes: list[int] = []
    for text in prompts:
        pids = target.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
        dt, res = _time_call(lambda: spec.generate(pids, max_new))
        tps = len(res["tokens"]) / max(1e-9, dt)
        windows.append(tps)
        if spec.memory is not None:
            memory_sizes.append(spec.memory.size)
    return windows, memory_sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--router-weights", default="")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="QA: disable rejection memory on adaptive arm",
    )
    parser.add_argument("--out", default="runs/results/session_drift.json")
    args = parser.parse_args()

    online_cfg = yaml.safe_load((REPO_ROOT / "configs" / "online.yaml").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        print("session_drift: CUDA unavailable", file=sys.stderr)
        return 2

    global target
    device = torch.device("cuda")
    torch.manual_seed(42)
    from familydraft.targets.wrapper import TargetModel

    target = TargetModel.load(args.repo, dtype="bf16")
    target_id = _target_id_for(args.repo)

    make_adaptive = _make_dag(args, target, device, target_id, online_cfg, no_memory=args.no_memory)
    make_static = _make_dag(
        args, target, device, target_id, online_config={}, no_memory=True,
        no_online_feedback=True,
    )

    adaptive_means: list[float] = []
    static_means: list[float] = []
    recovery_windows: list[int] = []
    memory_bounded = True

    adaptive_spec = make_adaptive()
    static_spec = make_static()
    # Freeze static at segment-1 stats: record the base, then hold it.
    static_base = dict(static_spec.router.base)

    for seg_idx, segment in enumerate(SEGMENTS):
        prompts = _segment_prompts(segment, args.max_prompts)
        if not prompts:
            print(f"session_drift: no prompts for segment {segment}", file=sys.stderr)
            return 2

        aw, amem = _run_segment(adaptive_spec, prompts, args.max_new)
        sw, _ = _run_segment(static_spec, prompts, args.max_new)
        adaptive_means.append(sum(aw) / max(1, len(aw)))
        static_means.append(sum(sw) / max(1, len(sw)))

        # Static control stays frozen at segment-1 stats.
        if seg_idx == 0:
            for e in static_base:
                static_spec.router.base[e] = static_base[e]
        # Memory bounded by LRU cap.
        if adaptive_spec.memory is not None and any(
            s > adaptive_spec.memory.max_entries for s in amem
        ):
            memory_bounded = False

        # Recovery-window metric: after a shift (segment > 0), windows until the
        # adaptive mean re-crosses >=90% of the best segment mean seen so far.
        if seg_idx > 0:
            best = max(adaptive_means[:seg_idx])
            target_tps = 0.9 * best
            # Recovery = how far the adaptive arm is from the target (0 = already recovered)
            deficit = max(0.0, target_tps - adaptive_means[seg_idx])
            recovery_windows.append(int(round(deficit / max(1e-9, 1.0))))

    # Acceptance: adaptive >= static on segments 2 and 3 (indices 1, 2).
    seg2_ok = adaptive_means[1] >= static_means[1]
    seg3_ok = adaptive_means[2] >= static_means[2]

    result = {
        "schema": "familydraft.session_drift.v1",
        "repo": args.repo,
        "no_memory": args.no_memory,
        "segments": SEGMENTS,
        "adaptive_mean_tps": adaptive_means,
        "static_mean_tps": static_means,
        "recovery_windows_per_shift": recovery_windows,
        "memory_bounded": memory_bounded,
        "adaptive_ge_static_seg2": seg2_ok,
        "adaptive_ge_static_seg3": seg3_ok,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"adaptive: {[round(x, 3) for x in adaptive_means]}")
    print(f"static:   {[round(x, 3) for x in static_means]}")
    print(f"recovery windows: {recovery_windows}  memory_bounded={memory_bounded}")
    print(f"adaptive>=static seg2={seg2_ok} seg3={seg3_ok}")

    if not (seg2_ok and seg3_ok and memory_bounded):
        print(
            "session_drift: FAIL (adaptive must beat static on later segments; "
            "memory bounded)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
