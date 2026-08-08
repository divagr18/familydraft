"""Multi-target trace campaign runner (plan todo 10).

Idempotent, resumable-per-shard greedy trace generation with top-k capture
(argmax + top-64 ids/logits/ranks per step, per docs/trace_format.md). Runs on
the pod (Qwen3-8B primary, 4B/14B secondary) but the script is locally
developed and config-hashed so the campaign is reproducible.

Resume: a shard that already has a MANIFEST.traces.json entry is skipped
(byte-identical MANIFEST entries on rerun). Budget guard: max GPU-hours per
config from configs/trace_campaign.yaml.

Usage (pod): python scripts/run_trace_campaign.py --traces-dir /workspace/traces
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from familydraft.targets.wrapper import TargetModel, topk_snapshot  # noqa: E402

TASK_CLASSES = ["chat", "code", "structured", "math"]
MANIFEST_FILE = {
    "chat": "mtbench/items.jsonl",
    "code": "humaneval/items.jsonl",
    "structured": "structured/items.jsonl",
    "math": "gsm8k/items.jsonl",
}


def _config_sha256(cfg: dict) -> str:
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prompts_for(task_class: str, max_prompts: int) -> list[str]:
    path = REPO_ROOT / "data" / "eval" / MANIFEST_FILE[task_class]
    items = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    prompts = [it.get("prompt_text") for it in items if it.get("prompt_text")]
    return prompts[:max_prompts] if max_prompts > 0 else prompts


def _generate_shard(
    target, prompts, max_new_tokens, topk, seed, target_id, cfg_sha
) -> list[dict]:
    """Greedy decode with per-step top-k capture (docs/trace_format.md schema)."""
    torch.manual_seed(seed)
    lines: list[dict] = []
    for text in prompts:
        pids = target.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        past = None
        x = pids.unsqueeze(0).to(target.device)
        with torch.inference_mode():
            out = target.model(input_ids=x, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        for step in range(max_new_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            snap = topk_snapshot(logits.unsqueeze(0), topk)
            chosen = int(snap.token_ids[0, 0])
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            lines.append({
                "step": step,
                "chosen_token": chosen,
                "topk": {
                    "ids": snap.token_ids[0].tolist(),
                    "logits": snap.logits[0].tolist(),
                    "ranks": snap.ranks[0].tolist(),
                },
                "latency_ms": latency_ms,
                "target_id": target_id,
                "config_sha256": cfg_sha,
                "seed": seed,
            })
            if chosen == target.tokenizer.eos_token_id:
                break
            with torch.inference_mode():
                o = target.model(
                    input_ids=torch.tensor([[chosen]], device=target.device),
                    past_key_values=past,
                    use_cache=True,
                )
            past = o.past_key_values
            logits = o.logits[0, -1]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", default="/workspace/traces")
    parser.add_argument("--config", default="configs/trace_campaign.yaml")
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B", help="local smoke target")
    parser.add_argument("--target-id", default="Qwen/Qwen3-8B", help="reported target_id")
    parser.add_argument("--local", action="store_true",
                        help="use local_run caps (dev-box smoke, not the pod campaign)")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="cap prompts per class (0 = all)")
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    cfg_sha = _config_sha256(cfg)
    topk = int(cfg["defaults"]["capture_topk"])
    seed = int(cfg["defaults"]["seed"])

    if args.local:
        max_new = int(cfg["local_run"]["max_new_tokens"])
        max_prompts = args.max_prompts or int(cfg["local_run"]["max_prompts_per_class"])
    else:
        max_new = int(cfg["max_new_tokens_by_class"]["code"])
        max_prompts = args.max_prompts or 0

    traces_dir = Path(args.traces_dir)
    manifest_path = traces_dir / "MANIFEST.traces.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    target = TargetModel.load(args.repo, dtype="bf16")

    total_hours = 0.0
    for task_class in TASK_CLASSES:
        shard_name = f"{task_class}.jsonl"
        shard_path = traces_dir / args.target_id / shard_name
        key = f"{args.target_id}/{shard_name}"
        if key in manifest and shard_path.exists():
            print(f"skip (resume): {key}", flush=True)
            continue

        prompts = _prompts_for(task_class, max_prompts)
        if not prompts:
            print(f"no prompts for {task_class}; skipping", flush=True)
            continue

        max_new_this = max_new
        if not args.local and task_class == "math":
            max_new_this = int(cfg["max_new_tokens_by_class"]["math"])

        t_start = time.perf_counter()
        lines = _generate_shard(
            target, prompts, max_new_this, topk, seed, args.target_id, cfg_sha
        )
        elapsed_hours = (time.perf_counter() - t_start) / 3600.0
        total_hours += elapsed_hours

        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with shard_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        manifest[key] = {
            "sha256": _sha256(shard_path),
            "steps": len(lines),
            "gpu_hours": round(elapsed_hours, 4),
            "task_class": task_class,
            "target_id": args.target_id,
        }
        print(f"wrote {key}: {len(lines)} steps in {elapsed_hours:.3f}h", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Budget guard.
    budget_hours = float(cfg["budget"]["max_gpu_hours_per_config"])
    if total_hours > budget_hours:
        print(f"budget exceeded: {total_hours:.2f}h > {budget_hours}h", file=sys.stderr)
        return 1
    print(f"campaign done: {len(manifest)} shards, {total_hours:.3f}h total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
