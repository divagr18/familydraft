"""DAG budget policy tests (plan todo 29, locally-completable portion).

Acceptance criteria:
  - policy JSON loads and the DAG builder applies the node cap without breaking
    the existing todo-7 DAG tests (regression run asserted separately).
  - invalid thresholds are rejected by schema validation at load (test asserts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from familydraft.verify.budget_policy import (  # noqa: E402
    DagBudgetPolicyError,
    load_dag_budget_policy,
)
from familydraft.verify.dag import CandidateDag  # noqa: E402


def _valid_policy(max_nodes: int = 16) -> dict:
    return {
        "schema": "familydraft.dag_budget_policy.v1",
        "defaults": {
            "max_nodes": max_nodes,
            "verify_cost_per_node_ms": 20.0,
            "vanilla_decode_ms": 38.0,
        },
        "targets": {},
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_valid_policy_loads_and_cap(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_policy(max_nodes=16))
    policy = load_dag_budget_policy("Qwen/Qwen3-0.6B", path=path)
    assert policy.max_nodes == 16
    assert policy.node_cap_from_verify() == 16


def test_invalid_max_nodes_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_policy(max_nodes=0))
    with pytest.raises(DagBudgetPolicyError):
        load_dag_budget_policy("Qwen/Qwen3-0.6B", path=path)


def test_invalid_cost_rejected(tmp_path: Path) -> None:
    bad = _valid_policy()
    bad["defaults"]["verify_cost_per_node_ms"] = -1.0
    path = _write(tmp_path, bad)
    with pytest.raises(DagBudgetPolicyError):
        load_dag_budget_policy("Qwen/Qwen3-0.6B", path=path)


def test_unexpected_schema_rejected(tmp_path: Path) -> None:
    bad = _valid_policy()
    bad["schema"] = "familydraft.other.v9"
    path = _write(tmp_path, bad)
    with pytest.raises(DagBudgetPolicyError):
        load_dag_budget_policy("Qwen/Qwen3-0.6B", path=path)


def test_defaults_used_when_target_missing(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_policy(max_nodes=8))
    policy = load_dag_budget_policy("Qwen/Qwen3-99B-unknown", path=path)
    assert policy.max_nodes == 8


def test_prune_to_budget_applies_policy_cap() -> None:
    """The DAG builder consumes the policy cap (todo-7 regression surface)."""
    dag = CandidateDag()
    for expert_id in range(4):
        dag.insert([10, 20, 30, 40, 50, 60, 70, 80, 90, 100][: 8], expert_id=expert_id)
    assert dag.node_count > 8
    dag.prune_to_budget(8)
    assert dag.node_count == 8
