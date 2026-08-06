"""Artifact check for the verification cost curve produced by bench_micro.py.

Skips when the artifact is absent (fresh clone without GPU runs); fails
when the artifact exists but is malformed or non-monotone beyond jitter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = Path("runs/microbench/cost_curve.json")
REQUIRED_FIELDS = {
    "schema",
    "repo",
    "gpu",
    "torch",
    "decode_ms_per_token",
    "verify_ms_by_nodes",
    "non_neural_expert_budget_ms",
    "config_sha256",
    "measured_utc",
}

pytestmark = pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/bench_micro.py first")


def test_cost_curve_schema_and_budget() -> None:
    record = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS - record.keys()
    assert not missing, f"cost curve missing fields: {sorted(missing)}"
    assert record["schema"] == "familydraft.cost_curve.v1"
    assert record["decode_ms_per_token"] > 0
    assert record["non_neural_expert_budget_ms"] == pytest.approx(
        0.1 * record["decode_ms_per_token"]
    )


def test_verify_latency_is_monotone_in_node_count() -> None:
    record = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    points = sorted(
        (int(nodes), ms) for nodes, ms in record["verify_ms_by_nodes"].items()
    )
    assert len(points) >= 5
    # allow 5% jitter between adjacent points, but the curve must trend up
    for (n0, ms0), (n1, ms1) in zip(points, points[1:]):
        assert ms1 >= ms0 * 0.95, f"verify latency dropped from {n0} to {n1} nodes"
    assert points[-1][1] >= points[0][1]
