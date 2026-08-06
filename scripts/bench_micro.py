"""Verification/draft latency microbenchmark (plan todo 9).

Measures on one device: target decode latency per token, DAG verification
latency by node count (v1 verifier = memoized prefix forwards), and the
derived latency budget for non-neural experts. Emits JSON consumed by the
router's utility objective and by tests/test_cost_curve.py.

The verification DAGs are built from the target's OWN greedy continuation so
acceptance walks the full depth — this is what makes verification cost scale
with node count instead of collapsing to a single prefix.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from familydraft.infra.run import config_fingerprint
from familydraft.targets.wrapper import TargetModel
from familydraft.verify.dag import CandidateDag
from familydraft.verify.dag_verifier import verify_dag_greedy

NODE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
DECODE_TOKENS = 256
WARMUP_DECODES = 3
REPEATS = 3


def _event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _bench_decode_ms_per_token(target: TargetModel, prompt_ids: torch.Tensor) -> float:
    for _ in range(WARMUP_DECODES):
        target.generate_greedy(prompt_ids, 16)
    samples = []
    for _ in range(REPEATS):
        samples.append(_event_ms(lambda: target.generate_greedy(prompt_ids, DECODE_TOKENS)))
    return statistics.median(samples) / DECODE_TOKENS


def _argmax_after(target: TargetModel, prompt_ids: torch.Tensor, prefix: tuple[int, ...]) -> int:
    ids = torch.cat([prompt_ids, torch.tensor(prefix, dtype=torch.long)])
    snap = target.topk_logits(ids.unsqueeze(0), k=1)
    return int(snap.token_ids[-1, 0])


def _dist_oracle(target: TargetModel, prompt_ids: torch.Tensor):
    """One-hot distribution oracle: the token the target greedily picks after a prefix."""
    vocab = target.vocab_size

    def dist_at(prefix: tuple[int, ...]) -> torch.Tensor:
        onehot = torch.zeros(vocab, dtype=torch.float32)
        onehot[_argmax_after(target, prompt_ids, prefix)] = 1.0
        return onehot

    return dist_at


def _self_consistent_continuation(
    target: TargetModel, prompt_ids: torch.Tensor, length: int
) -> list[int]:
    """Greedy continuation produced by the SAME full-forward path the verifier
    uses for scoring. Each token is, by construction, the argmax of the forward
    over prompt+prefix, so acceptance walks the full depth. This avoids the
    bf16 incremental-decode vs full-forward argmax drift that would otherwise
    truncate acceptance (and flatten the cost curve) at ~16 tokens.
    """
    prefix: list[int] = []
    for _ in range(length):
        prefix.append(_argmax_after(target, prompt_ids, tuple(prefix)))
    return prefix


def _bench_verify_ms(
    target: TargetModel, prompt_ids: torch.Tensor, node_count: int, continuation: list[int]
) -> float:
    """Verify a chain DAG of depth=node_count built from accepted tokens.

    Because the continuation is the target's own greedy output, every drafted
    token matches the target argmax, acceptance walks the full depth, and the
    verifier forwards every distinct prefix (node_count + bonus). Verification
    latency therefore grows with node count, which is the quantity the router's
    cost model needs.
    """
    depth = max(1, min(node_count, len(continuation)))
    dag = CandidateDag()
    dag.insert(continuation[:depth], expert_id=0)
    dist_at = _dist_oracle(target, prompt_ids)
    verify_dag_greedy(dag, dist_at)  # warm kernels for this depth
    samples = []
    for _ in range(REPEATS):
        samples.append(_event_ms(lambda: verify_dag_greedy(dag, dist_at)))
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Speculative decoding works by")
    parser.add_argument("--out", default="runs/microbench/cost_curve.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("bench_micro: CUDA GPU unavailable; refusing to run on CPU.", file=sys.stderr)
        return 2

    config = {"repo": args.repo, "prompt": args.prompt, "node_counts": list(NODE_COUNTS)}
    target = TargetModel.load(args.repo, dtype="bf16")
    prompt_ids = target.tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]

    max_nodes = max(NODE_COUNTS)
    continuation = _self_consistent_continuation(target, prompt_ids, max_nodes)

    decode_ms = _bench_decode_ms_per_token(target, prompt_ids)
    verify_by_nodes = {
        str(n): _bench_verify_ms(target, prompt_ids, n, continuation) for n in NODE_COUNTS
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "familydraft.cost_curve.v1",
        "repo": args.repo,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "bf16",
        "platform": platform.platform(),
        "decode_ms_per_token": decode_ms,
        "verify_ms_by_nodes": verify_by_nodes,
        "non_neural_expert_budget_ms": 0.1 * decode_ms,
        "config_sha256": config_fingerprint(config),
        "measured_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
