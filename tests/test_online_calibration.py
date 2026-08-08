"""Online calibration tests (plan todo 26).

Acceptance criteria:
  1. Agreement rule extends the draft horizon on a crafted two-expert agreement
     case (asserted horizon change vs no agreement).
  2. Isotonic calibration is monotone (predicted order preserved) on synthetic
     skewed data.
  3. Abstention ROC emits an AUC field from a dev-style score list.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from familydraft.calibration import (  # noqa: E402
    IsotonicCalibration,
    abstention_roc,
    agreement_extension,
    agreement_stats,
)
from familydraft.verify.dag import CandidateDag  # noqa: E402


def _two_expert_agreeing_dag():
    """Two experts whose proposals share a long common prefix (agreement)."""
    dag = CandidateDag()
    dag.insert([10, 20, 30, 40, 50], expert_id=0)
    dag.insert([10, 20, 30, 60, 70], expert_id=1)
    return dag


def _two_expert_disagreeing_dag():
    """Two experts whose proposals diverge immediately (no agreement)."""
    dag = CandidateDag()
    dag.insert([10, 20, 30, 40, 50], expert_id=0)
    dag.insert([99, 98, 97, 96, 95], expert_id=1)
    return dag


def test_agreement_rule_extends_horizon() -> None:
    agree = agreement_stats(_two_expert_agreeing_dag(), min_agree=2)
    disagree = agreement_stats(_two_expert_disagreeing_dag(), min_agree=2)

    assert agree.agreeing_nodes > 0, "agreement DAG should have agreeing nodes"
    assert disagree.agreeing_nodes == 0, "disagreement DAG should have none"
    assert agree.agreement_fraction > disagree.agreement_fraction

    base_horizon = 4
    max_horizon = 8
    h_agree = agreement_extension(agree, base_horizon, max_horizon)
    h_disagree = agreement_extension(disagree, base_horizon, max_horizon)
    assert h_agree > h_disagree, (
        f"agreement must extend horizon: agree={h_agree} disagree={h_disagree}"
    )
    assert h_agree <= max_horizon


def test_isotonic_calibration_is_monotone() -> None:
    """Predicted order must be preserved after calibration on skewed data."""
    cal = IsotonicCalibration(window=64)
    # Synthetic skewed pairs: predicted rises but measured is noisy/skewed.
    for i in range(40):
        pred = i / 40.0
        measured = pred + (0.15 if pred > 0.7 else 0.0)  # upward skew at high pred
        cal.update(pred, measured)

    ordered_preds = [0.05, 0.25, 0.5, 0.75, 0.95]
    calibrated = [cal.calibrate(p) for p in ordered_preds]
    for a, b in zip(calibrated, calibrated[1:]):
        assert a <= b + 1e-9, f"calibration broke monotonicity: {calibrated}"


def test_abstention_roc_emits_auc() -> None:
    scores = [(0.9, 1), (0.8, 1), (0.7, 0), (0.6, 1), (0.5, 0), (0.4, 0), (0.3, 1), (0.2, 0)]
    roc = abstention_roc(scores)
    assert "auc" in roc, "ROC must emit an AUC field"
    assert 0.0 <= roc["auc"] <= 1.0
    assert len(roc["thresholds"]) == len(scores)
    assert len(roc["fpr"]) == len(roc["tpr"])


def test_empty_roc_is_safe() -> None:
    roc = abstention_roc([])
    assert roc["auc"] == 0.0
