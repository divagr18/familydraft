"""Target-variant zero-shot transfer evaluation harness (plan todo 24).

Runs the distillation-trained drafter (general expert) against target variants
and records accepted-length / tokens-per-sec / drafter-overhead per (target,
task class). The seen-target GUARD refuses to evaluate a target that was in the
training set (QA requirement: evaluating a seen target would defeat the
zero-shot transfer test). Emits runs/results/phase2_transfer.csv plus a
transfer-delta log (unseen vs seen baseline per task class).

Per the plan: train on {4B, 8B, 14B} only, evaluate zero-shot transfer on
UNSEEN targets (32B, Coder-30B-A3B) without per-target retraining.

The full-matrix training (8B/14B/32B/Coder-30B-A3B) is pod-deferred; this
harness is locally runnable for a smoke transfer test on unseen targets that
fit the dev box (e.g. 0.6B, which is not in the training set), and on the pod
for the real matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_CLASSES = ["code", "structured"]
MANIFEST_FILE = {
    "code": "humaneval/items.jsonl",
    "structured": "structured/items.jsonl",
}
# Training set per the plan: the drafter is trained on these target ids only.
TRAIN_TARGETS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
CSV_COLUMNS = [
    "target", "task_class", "accepted_length", "tokens_per_sec",
    "drafter_overhead_rounds", "config_hash", "seed",
]


def _target_id_for(repo: str) -> int:
    table_path = REPO_ROOT / "configs" / "target_ids.json"
    if table_path.exists():
        table = json.loads(table_path.read_text(encoding="utf-8"))
        if repo in table:
            return int(table[repo]["id"])
    return 0


def _time_call(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _prompts_for(task_class: str, max_prompts: int) -> list[str]:
    path = REPO_ROOT / "data" / "eval" / MANIFEST_FILE[task_class]
    items = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    prompts = [it.get("prompt_text") for it in items if it.get("prompt_text")]
    return prompts[:max_prompts] if max_prompts > 0 else prompts


def _config_hash(*payloads) -> str:
    h = hashlib.sha256()
    for p in payloads:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def _chain_factory(args, target, device, target_id):
    """Fresh IntegratedSpeculator with a fresh KV-cache drafter (matches
    run_baselines._chain_factory so transfer rows are comparable)."""
    from familydraft.draft.trunk import build_trunk_from_config
    from familydraft.eval.draft_loop import IntegratedSpeculator, make_general_drafter
    from familydraft.experts.general import GeneralExpert

    trunk = build_trunk_from_config(REPO_ROOT).to(device)
    expert = GeneralExpert(trunk).to(device)
    if args.general_checkpoint:
        expert.load_state_dict(torch.load(args.general_checkpoint, map_location=device))
    drafter = make_general_drafter(expert, args.spec_len, target_id, device)

    def make():
        return IntegratedSpeculator(target, drafter, args.spec_len, target_id)

    return make


def _validate_eval_targets(eval_targets: list[str]) -> list[str]:
    """QA guard: refuse to evaluate a training-set target (would defeat the
    zero-shot transfer test). Returns the cleaned list or raises."""
    seen = [t for t in eval_targets if t in TRAIN_TARGETS]
    if seen:
        raise ValueError(
            f"run_transfer_eval: REFUSED - seen target(s) in eval set: {seen} "
            f"(training set = {TRAIN_TARGETS})"
        )
    return [t for t in eval_targets if t]


def _compute_delta(rows: list[dict], seen_baseline_path: Path | None) -> list[dict]:
    """Transfer delta (unseen vs seen) per task class, logged to
    runs/results/phase2_transfer_delta.csv."""
    if seen_baseline_path is None or not seen_baseline_path.exists():
        return []
    seen = {}
    with seen_baseline_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            seen[r["task_class"]] = float(r["accepted_length"])
    deltas = []
    for r in rows:
        tc = r["task_class"]
        if tc in seen:
            deltas.append({
                "target": r["target"],
                "task_class": tc,
                "unseen_accepted_length": r["accepted_length"],
                "seen_accepted_length": round(seen[tc], 3),
                "transfer_delta": round(r["accepted_length"] - seen[tc], 3),
            })
    return deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-targets", default="Qwen/Qwen3-0.6B",
                        help="comma list; must NOT overlap the training set {4B,8B,14B}")
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--general-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seen-baseline", default="runs/results/phase2_train_eval.csv",
                        help="seen-target baseline CSV (accepted_length per task_class); "
                             "optional, enables transfer-delta logging")
    parser.add_argument("--out", default="runs/results/phase2_transfer.csv")
    parser.add_argument("--delta-out", default="runs/results/phase2_transfer_delta.csv")
    args = parser.parse_args()

    eval_targets = [t.strip() for t in args.eval_targets.split(",") if t.strip()]
    try:
        eval_targets = _validate_eval_targets(eval_targets)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("run_transfer_eval: CUDA unavailable", file=sys.stderr)
        return 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    rows = []
    for repo in eval_targets:
        from familydraft.targets.wrapper import TargetModel

        target_id = _target_id_for(repo)
        target = TargetModel.load(repo, dtype="bf16")
        make_spec = _chain_factory(args, target, device, target_id)
        cfg_hash = _config_hash({
            "repo": repo, "target_id": target_id, "spec_len": args.spec_len,
            "max_new": args.max_new, "max_prompts": args.max_prompts,
            "checkpoint": args.general_checkpoint, "seed": args.seed,
        })

        for task_class in TASK_CLASSES:
            prompts = _prompts_for(task_class, args.max_prompts)
            accepted_total = 0
            times = []
            overhead_total = 0
            for text in prompts:
                spec = make_spec()
                pids = target.tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0].tolist()
                dt, res = _time_call(lambda: spec.generate(pids, args.max_new))
                times.append(dt)
                accepted_total += res["accepted_tokens"]
                overhead_total += res["rounds"]
            tps = args.max_new / max(1e-9, sum(times))
            rows.append({
                "target": repo,
                "task_class": task_class,
                "accepted_length": round(accepted_total / max(1, len(prompts)), 3),
                "tokens_per_sec": round(tps, 3),
                "drafter_overhead_rounds": round(overhead_total / max(1, len(prompts)), 3),
                "config_hash": cfg_hash,
                "seed": args.seed,
            })

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    deltas = _compute_delta(rows, REPO_ROOT / args.seen_baseline)
    if deltas:
        delta_path = REPO_ROOT / args.delta_out
        with delta_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "target", "task_class", "unseen_accepted_length",
                "seen_accepted_length", "transfer_delta"])
            writer.writeheader()
            writer.writerows(deltas)
        print(
            f"run_transfer_eval: wrote {delta_path.relative_to(REPO_ROOT)} "
            f"({len(deltas)} deltas)"
        )

    print(f"run_transfer_eval: wrote {out_path.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['target']}/{r['task_class']}: acc_len={r['accepted_length']} "
              f"tps={r['tokens_per_sec']} rounds={r['drafter_overhead_rounds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
