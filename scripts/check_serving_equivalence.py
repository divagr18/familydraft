"""SGLang-vs-harness output equivalence checker (plan todo 28, locally-completable).

The plan's Phase-4 acceptance requires: "output equivalence check: 50 prompts
greedy — SGLang-integrated outputs must be IDENTICAL to standalone harness
outputs", and the QA failure scenario seeds a deliberate divergence and
requires the check to fail NAMING divergent prompt ids.

This harness compares two greedy-output files (one per system) and asserts
byte-identical tokens per prompt. `--seed-divergence` deliberately corrupts
one prompt's output so the QA failure path is provably exercised.

The real SGLang-vs-standalone comparison runs on the pod (todo 28 execution);
the harness + QA divergence test are locally-runnable.

Output files are JSONL: {"id": <prompt id>, "tokens": [int, ...]}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_outputs(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[str(rec["id"])] = [int(t) for t in rec["tokens"]]
    return out


def compare(reference: dict[str, list[int]], candidate: dict[str, list[int]]) -> list[str]:
    """Return list of divergent prompt ids (empty == 50/50 identical)."""
    divergent: list[str] = []
    for pid, ref_tokens in reference.items():
        cand_tokens = candidate.get(pid)
        if cand_tokens != ref_tokens:
            divergent.append(pid)
    return divergent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True,
                        help="standalone-harness greedy outputs (JSONL: id, tokens)")
    parser.add_argument("--candidate", required=True,
                        help="SGLang-integrated greedy outputs (JSONL: id, tokens)")
    parser.add_argument("--seed-divergence", action="store_true",
                        help="QA mode: corrupt one candidate output to prove the check fails")
    parser.add_argument("--min-prompts", type=int, default=50,
                        help="prompts the equivalence check must cover (plan: 50)")
    args = parser.parse_args()

    ref = load_outputs(Path(args.reference))
    cand = load_outputs(Path(args.candidate))

    if len(ref) < args.min_prompts or len(cand) < args.min_prompts:
        print(f"check_serving_equivalence: coverage insufficient - reference {len(ref)} "
              f"candidate {len(cand)}, required {args.min_prompts}", file=sys.stderr)
        return 2

    if args.seed_divergence:
        pid = next(iter(ref))
        cand[pid] = [0, 1, 2, 3]  # deliberate wrong-token corruption
        print(f"check_serving_equivalence: seeded divergence at prompt {pid}")

    divergent = compare(ref, cand)
    if divergent:
        print(f"check_serving_equivalence: FAIL - {len(divergent)} divergent prompt(s): "
              f"{sorted(divergent)[:10]}", file=sys.stderr)
        return 1
    print(f"check_serving_equivalence: PASS - {len(ref)}/{len(ref)} prompts identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
