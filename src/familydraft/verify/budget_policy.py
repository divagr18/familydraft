"""DAG budget policy loader with schema validation (plan todo 29).

Loads configs/dag_budget_policy.json and validates it at load time: per-target
node budgets must be positive integers, costs must be positive floats, and the
cap rule (verify_cost(m)*m^-1 > vanilla_decode -> cap) is enforced by the DAG
builder. Invalid thresholds raise DagBudgetPolicyError so a malformed policy is
rejected loudly instead of silently mis-capping the DAG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DagBudgetPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class DagBudgetPolicy:
    max_nodes: int
    verify_cost_per_node_ms: float
    vanilla_decode_ms: float

    def node_cap_from_verify(self) -> int:
        """Cap where verify_cost(m)*m^-1 exceeds vanilla decode cost.

        The largest m such that verify_cost_per_node_ms * m / m <= vanilla
        never holds once per-node cost exceeds vanilla; we cap at the smallest
        budget where the marginal node is no longer worth verifying. Per the
        plan: cap nodes where verify_cost(m)*m^-1 exceeds vanilla decode.
        """
        if self.verify_cost_per_node_ms >= self.vanilla_decode_ms:
            return 1  # even one extra node is not worth verifying
        return self.max_nodes


def _validate_section(section: dict, where: str) -> DagBudgetPolicy:
    max_nodes = section.get("max_nodes")
    verify = section.get("verify_cost_per_node_ms")
    vanilla = section.get("vanilla_decode_ms")
    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise DagBudgetPolicyError(
            f"{where}.max_nodes must be a positive integer, got {max_nodes!r}"
        )
    for key, val in (("verify_cost_per_node_ms", verify), ("vanilla_decode_ms", vanilla)):
        if not isinstance(val, (int, float)) or val <= 0:
            raise DagBudgetPolicyError(f"{where}.{key} must be a positive number, got {val!r}")
    return DagBudgetPolicy(int(max_nodes), float(verify), float(vanilla))


def load_dag_budget_policy(
    repo: str, path: Path | None = None
) -> DagBudgetPolicy:
    policy_path = path or (REPO_ROOT / "configs" / "dag_budget_policy.json")
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DagBudgetPolicyError(f"unreadable policy file {policy_path}: {exc}") from exc

    if raw.get("schema") != "familydraft.dag_budget_policy.v1":
        raise DagBudgetPolicyError(f"unexpected policy schema: {raw.get('schema')!r}")

    defaults = _validate_section(raw.get("defaults", {}), "defaults")
    targets = raw.get("targets", {})
    if not isinstance(targets, dict):
        raise DagBudgetPolicyError("targets must be an object keyed by repo id")
    target_section = targets.get(repo)
    if target_section is None:
        return defaults
    merged = {**raw["defaults"], **target_section}
    return _validate_section(merged, f"targets.{repo}")


def _policy_for(repo: str) -> DagBudgetPolicy:
    return load_dag_budget_policy(repo)
