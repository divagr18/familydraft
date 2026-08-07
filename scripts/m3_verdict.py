"""M3 verdict computation (plan todo 22).

Reads `runs/results/phase1.csv` and computes the Phase-1 verdict against the
PRE-REGISTERED `configs/verdict_protocol.yaml` thresholds. Exits 0 on PASS,
78 on FAIL (matching the plan's gate exit codes), or 2 if inputs are missing.

Verdict rules (from the protocol):
  - every greedy row must have fp32_exact=true (or bf16 artifact documented)
  - full_proposal_moe must beat vanilla_ar on every task class
  - full_proposal_moe must beat equal_flop_dense_drafter on every task class
  - full_proposal_moe must not lose to single_best_expert on more than 1 class
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

PROTOCOL = Path("configs/verdict_protocol.yaml")
CSV = Path("runs/results/phase1.csv")


def _load_rows() -> list[dict]:
    if not CSV.exists():
        print(f"m3_verdict: missing {CSV}", file=sys.stderr)
        return []
    with CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _row(rows, system: str, task_class: str) -> dict | None:
    for r in rows:
        if r.get("system") == system and r.get("task_class") == task_class:
            return r
    return None


def _speedup(dag_tps: float, base_tps: float) -> float:
    return dag_tps / max(1e-9, base_tps)


def main() -> int:
    if not PROTOCOL.exists():
        print(f"m3_verdict: missing protocol {PROTOCOL}", file=sys.stderr)
        return 2
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    task_classes = protocol["task_classes"]
    rows = _load_rows()
    if not rows:
        return 2

    failures: list[str] = []

    # 1. exactness invariant on every row
    for r in rows:
        fp32 = r.get("exactness_fp32_exact", "").lower() == "true"
        bf16 = float(r.get("exactness_bf16_exact_match_rate", "0") or 0)
        if not fp32 and bf16 < 0.9:
            failures.append(
                f"{r.get('system')}/{r.get('task_class')}: fp32_exact=false and "
                f"bf16 match={bf16:.2f} < 0.9"
            )

    # 2. DAG vs vanilla on every task class
    for tc in task_classes:
        dag = _row(rows, "full_proposal_moe", tc)
        vanilla = _row(rows, "vanilla_ar", tc)
        if dag is None or vanilla is None:
            failures.append(f"missing full_proposal_moe or vanilla_ar row for {tc}")
            continue
        sp = _speedup(float(dag["target_tokens_per_second"]),
                      float(vanilla["target_tokens_per_second"]))
        if sp <= protocol["verdict"]["min_speedup_vs_vanilla"]:
            min_sp = protocol["verdict"]["min_speedup_vs_vanilla"]
            failures.append(f"{tc}: DAG speedup {sp:.3f} <= {min_sp}")

    # 3. DAG vs equal-FLOP dense drafter
    if protocol["verdict"]["must_beat_equal_flop_dense"]:
        for tc in task_classes:
            dag = _row(rows, "full_proposal_moe", tc)
            ef = _row(rows, "equal_flop_dense_drafter", tc)
            if dag is None or ef is None:
                failures.append(f"missing equal_flop_dense_drafter row for {tc}")
                continue
            sp = _speedup(float(dag["target_tokens_per_second"]),
                          float(ef["target_tokens_per_second"]))
            if sp < 1.0:
                failures.append(f"{tc}: DAG < equal-FLOP dense ({sp:.3f}x)")

    # 4. DAG vs single-best-expert
    max_losses = protocol["verdict"]["max_losses_vs_single_best"]
    losses = 0
    for tc in task_classes:
        dag = _row(rows, "full_proposal_moe", tc)
        sbe = _row(rows, "single_best_expert", tc)
        if dag is None or sbe is None:
            failures.append(f"missing single_best_expert row for {tc}")
            continue
        if float(dag["target_tokens_per_second"]) < float(sbe["target_tokens_per_second"]):
            losses += 1
    if losses > max_losses:
        failures.append(f"DAG loses to single-best-expert on {losses} classes (max {max_losses})")

    if failures:
        print("M3 VERDICT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 78
    print("M3 VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
