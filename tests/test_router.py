"""Utility router v1 tests (plan todo 19)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from familydraft.router.router import (
    UtilityRouter,
    make_features,
)

VERIFY_CURVE = {2: 40.0, 66: 1320.0}  # marginal verify cost = 20.0/node


def _router(**kwargs):
    defaults = dict(
        expert_names=["general", "macro", "copy", "reject_memory"],
        draft_ms={"general": 10.0, "macro": 1.0, "copy": 0.5, "reject_memory": 0.5},
        verify_ms_by_nodes=VERIFY_CURVE,
        tau_abstain=0.01,
        ema_rate=0.5,
    )
    defaults.update(kwargs)
    return UtilityRouter(**defaults)


def test_verify_marginal_cost_hand_computed() -> None:
    r = _router()
    assert r.verify_ms_per_node == pytest.approx(20.0, abs=1e-9)


def test_utility_matches_hand_computed_golden() -> None:
    r = _router()
    r.base["general"] = 2.0
    features = make_features(0.0, 0.0, 0.0, 0.0, target_id=0)
    horizon = r.horizon_for("general")
    assert horizon == 2
    u = r.utility("general", features, horizon)
    expected = 2.0 / (10.0 + 20.0 * 2)
    assert u == pytest.approx(expected, abs=1e-9)


def test_select_returns_valid_decision() -> None:
    r = _router()
    features = make_features(0.5, 0.5, 0.1, 0.2, target_id=2)
    decision = r.select(features, max_experts=2)
    assert decision.abstain is False
    assert 1 <= len(decision.expert_subset) <= 2
    for e in decision.expert_subset:
        assert e in r.expert_names
        assert decision.horizons[e] in (2, 4, 6, 8)


def test_low_utility_input_abstains() -> None:
    r = _router(tau_abstain=10.0)
    r.base = {e: 0.0 for e in r.expert_names}
    features = make_features(0.0, 0.0, 0.0, 0.0, target_id=0)
    decision = r.select(features)
    assert decision.abstain is True
    assert decision.expert_subset == []


def test_ema_update_is_deterministic_golden() -> None:
    r = _router(ema_rate=0.5)
    r.update_feedback("copy", accepted_len=3.0, draft_ms=0.5, first_rejection=2.0)
    assert r.stats["copy"].accepted_len_ema == pytest.approx(2.0, abs=1e-9)
    r.update_feedback("copy", accepted_len=1.0, draft_ms=0.5, first_rejection=1.0)
    assert r.stats["copy"].accepted_len_ema == pytest.approx(1.5, abs=1e-9)


def test_marginal_utility_prefers_least_overlap() -> None:
    r = _router()
    r.base = {"general": 2.0, "macro": 1.0, "copy": 1.0, "reject_memory": 1.0}
    r.record_overlap("general", "copy", 0.9)
    r.record_overlap("general", "macro", 0.95)
    r.record_overlap("general", "reject_memory", 0.1)
    features = make_features(0.0, 0.0, 0.0, 0.0, target_id=0)
    decision = r.select(features, max_experts=2)
    assert decision.expert_subset[0] == "general"
    assert "copy" not in decision.expert_subset[1:] or "reject_memory" in decision.expert_subset
    assert decision.expert_subset[1] == "reject_memory"


def test_latency_spike_switches_selection() -> None:
    r = _router(expert_names=["general", "macro"], draft_ms={"general": 1.0, "macro": 1.0})
    r.base = {"general": 1.0, "macro": 1.0}
    features = make_features(0.0, 0.0, 0.0, 0.0, target_id=0)
    before = r.select(features, max_experts=1)
    assert before.expert_subset[0] == "general"
    r.update_feedback("general", accepted_len=1.0, draft_ms=1000.0, first_rejection=1.0)
    after = r.select(features, max_experts=1)
    assert after.expert_subset[0] == "macro"


def test_cold_start_sets_base_from_simulated_acceptance() -> None:
    r = _router()
    r.cold_start({"general": 12.0, "macro": 1.2, "copy": 2.0, "reject_memory": 0.4})
    assert r.base["general"] == pytest.approx(12.0)
    # cold-start seeds accepted_len_ema at 1.0 so selection is by base quality,
    # not an inflated horizon that makes high-base experts look expensive.
    assert r.stats["general"].accepted_len_ema == pytest.approx(1.0)
    features = make_features(0.0, 0.0, 0.0, 0.0, target_id=0)
    decision = r.select(features, max_experts=1)
    assert decision.expert_subset[0] == "general"


def test_config_file_matches_defaults() -> None:
    cfg_path = Path(__file__).parent.parent / "configs" / "router.yaml"
    cfg = yaml.safe_load(cfg_path.read_text("utf-8"))
    assert cfg["horizons"] == [2, 4, 6, 8]
    assert cfg["max_experts"] == 2
    assert cfg["experts"] == ["general", "macro", "copy", "reject_memory"]
