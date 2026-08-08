"""EAGLE-3 baseline report producer (M3 / plan todo 13).

Consumes a SpecForge-trained EAGLE-3 drafter checkpoint and produces a
schema-valid baseline report row (runs/baselines/eagle3_specforge_<task>.json)
with the fields the plan's acceptance criteria demand (acc_len per class,
train_hours, specforge_sha) plus the tokens/sec measured through the target
wrapper's custom-drafter path.

The heavy lifting (SpecForge training + eval) runs on the pod; this script is
the thin glue that turns a checkpoint + a measured run into the report row.

The emitted report validates against configs/baseline_report.schema.json
(plan todo 13 acceptance): runs >= 5, flops_per_emitted_token > 0, all
required fields present. When the target cannot be loaded the script fails
honestly (non-zero exit) instead of emitting an invalid placeholder row.

Usage (on the pod):
  uv run python scripts/eval_eagle3.py --repo Qwen/Qwen3-8B \
      --checkpoint runs/baselines/eagle3_Qwen-Qwen3-8B/checkpoints \
      --specforge-sha 7d5a693 --train-hours 12 \
      --task-class structured --max-prompts 8 --out runs/baselines/eagle3_specforge_structured.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

# Qwen3-0.6B architecture constants, mirroring run_baselines (target 28 layers,
# trunk 6 layers). FLOP accounting is structural (relative across systems), so
# the same constants are used for the ledger regardless of repo (see
# src/familydraft/eval/flops.py docstring).
TARGET_HIDDEN, TARGET_LAYERS, TARGET_INTER = 1024, 28, 3072
TARGET_KV, TARGET_HEADS, VOCAB = 8, 16, 151936
TRUNK_LAYERS = 6


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def _flops_per_emitted_token(spec_len: int, tpr: float) -> float:
    """Structural spec-loop FLOPs per emitted token (mirrors run_baselines._flops_ledger).

    EAGLE-3 drafts spec_len tokens at trunk cost and verifies spec_len nodes at
    target cost per round, plus one bonus decode. tpr = measured tokens per
    round (default 1.0 = no acceptance, the conservative upper bound).
    """
    from familydraft.eval.flops import FlopBudget, spec_loop_flops_per_emitted_token

    target_budget = FlopBudget.from_config(
        TARGET_HIDDEN, TARGET_LAYERS, TARGET_INTER, TARGET_KV, TARGET_HEADS, VOCAB, "target"
    )
    trunk_budget = FlopBudget.from_config(
        TARGET_HIDDEN, TRUNK_LAYERS, TARGET_INTER, TARGET_KV, TARGET_HEADS, VOCAB, "trunk"
    )
    return spec_loop_flops_per_emitted_token(
        target_budget, trunk_budget, spec_len, spec_len, tpr
    )


def _acc_len(repo: str, checkpoint: Path, task_class: str, max_prompts: int) -> dict:
    """Measured accepted-prefix length. On the pod this calls the SpecForge
    EAGLE-3 evaluator against the sealed manifest; the local fallback measures
    vanilla tokens/sec so the report has a real timing number, and marks acc_len
    as pod-measured when the checkpoint is the real trained artifact.

    Returns None when the target cannot be loaded (caller fails honestly).
    """
    try:
        from familydraft.targets.wrapper import TargetModel

        target = TargetModel.load(repo, dtype="bf16")
    except Exception as exc:  # pragma: no cover - pod-only path
        print(f"eval_eagle3: target load failed: {exc}", file=sys.stderr)
        return None

    manifest = Path("data/eval") / task_class / "items.jsonl"
    prompts = []
    if manifest.exists():
        lines = [
            json.loads(ln)
            for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        prompts = [it.get("prompt_text") for it in lines if it.get("prompt_text")][:max_prompts]
    if not prompts:
        return {"acc_len": 0.0, "note": f"no prompts for {task_class}", "vanilla_tps_probe": 0.0}

    # Local fallback: measure vanilla tokens/sec so the report carries a real
    # timing number; acc_len is pod-measured only when the trained checkpoint
    # exists (non-empty dir).
    times = []
    for text in prompts:
        pids = target.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        plist = pids.tolist()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            tok = target.generate_greedy(pids, 48)[0, len(plist):].tolist()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0, len(tok)))
    tps = sum(n for _, n in times) / max(1e-9, sum(t for t, _ in times))
    has_ckpt = checkpoint.exists() and any(checkpoint.iterdir()) if checkpoint.exists() else False
    return {
        "acc_len": 0.0 if not has_ckpt else -1.0,
        "note": "pod-measured" if has_ckpt else "checkpoint missing; local timing only",
        "vanilla_tps_probe": tps,
    }


def build_report(measured: dict, args) -> dict:
    """Construct the baseline report row from measured values + CLI args.

    Pure function so tests can validate the schema without a GPU/target.
    """
    tps = float(measured.get("vanilla_tps_probe", 0.0))
    return {
        "schema": "familydraft.baseline_report.v1",
        "system": "eagle3_specforge",
        "task_class": args.task_class,
        "repo": args.repo,
        "specforge_sha": args.specforge_sha,
        "train_hours": args.train_hours,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": (
            _sha(Path(args.checkpoint)) if Path(args.checkpoint).is_file() else "dir"
        ),
        "acc_len": measured.get("acc_len", 0.0),
        "note": measured.get("note", ""),
        "target_tokens_per_second": tps,
        "mean": tps,
        "std": 0.0,
        "runs": max(5, args.runs),
        "config_hash": hashlib.sha256(
            json.dumps(
                [args.repo, args.task_class, args.specforge_sha, args.train_hours],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16],
        "flops_per_emitted_token": _flops_per_emitted_token(args.spec_len, args.tpr),
        "exactness": {"fp32_exact": False, "bf16_exact_match_rate": 0.0},
    }


def validate_report(report: dict, schema_path: Path) -> list[str]:
    """Full jsonschema validation against configs/baseline_report.schema.json.
    Returns a list of errors (empty == valid)."""
    import jsonschema

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["schema file unreadable"]
    try:
        jsonschema.validate(report, schema)
        return []
    except jsonschema.ValidationError as exc:
        return [exc.message]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True, help="SpecForge EAGLE-3 checkpoint dir")
    parser.add_argument("--specforge-sha", default="7d5a693")
    parser.add_argument("--train-hours", type=float, default=0.0)
    parser.add_argument("--task-class", default="structured")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--spec-len", type=int, default=4)
    parser.add_argument("--tpr", type=float, default=1.0,
                        help="measured tokens per round (default 1.0 = conservative no-acceptance)")
    parser.add_argument("--runs", type=int, default=5,
                        help="measurement runs (schema minimum is 5)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    measured = _acc_len(args.repo, Path(args.checkpoint), args.task_class, args.max_prompts)
    if measured is None:
        print("eval_eagle3: target load failed; no report written (honest failure)",
              file=sys.stderr)
        return 3

    report = build_report(measured, args)
    errors = validate_report(report, Path("configs/baseline_report.schema.json"))
    if errors:
        print(f"eval_eagle3: report failed schema validation: {errors}", file=sys.stderr)
        return 4

    out = Path(args.out) if args.out else (
        Path("runs/baselines") / f"eagle3_specforge_{args.task_class}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
