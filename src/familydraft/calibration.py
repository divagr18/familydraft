"""Online calibration utilities (plan todo 26).

Implements the three §6.3 / §7.5 calibration mechanisms WITHOUT touching model
weights:
  1. Agreement rule: nodes supported by >=2 experts are agreement points; the
     DAG can extend the speculative horizon and prioritise the agreed prefix.
  2. Isotonic calibration: pool-adjacent-violators (PAV) fit over a rolling
     window of (predicted acceptance, measured acceptance) so the router's
     utility estimates stay monotone and calibrated.
  3. Abstention ROC: given (predicted utility, did-speculation-beat-vanilla)
     pairs over a dev set, emit a precision/recall curve + AUC so the abstain
     threshold can be set honestly.

No weight updates; per-target scoping is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgreementStats:
    """Per-DAG agreement summary (concept note §6.3)."""
    agreeing_nodes: int = 0
    total_nodes: int = 0
    max_agreement: int = 0  # max experts agreeing on any single node

    @property
    def agreement_fraction(self) -> float:
        return self.agreeing_nodes / max(1, self.total_nodes)


def agreement_stats(dag, min_agree: int = 2) -> AgreementStats:
    """Count nodes supported by >= `min_agree` experts (an agreement point).

    The caller uses this to extend the horizon / reduce branching on the
    agreed prefix. Pure function over the DAG trie; no mutation.
    """
    stats = AgreementStats()
    for node in dag.nodes_topo():
        if node.node_id == 0:
            continue
        stats.total_nodes += 1
        support = len(node.experts)
        stats.max_agreement = max(stats.max_agreement, support)
        if support >= min_agree:
            stats.agreeing_nodes += 1
    return stats


def agreement_extension(stats: AgreementStats, base_horizon: int, max_horizon: int) -> int:
    """Horizon extension from agreement (§6.3): an agreed prefix earns up to
    +2 tokens, scaled by how many nodes agreed relative to the tree."""
    if stats.agreement_fraction <= 0:
        return base_horizon
    boost = int(round(2 * stats.agreement_fraction))
    return min(max_horizon, base_horizon + boost)


class IsotonicCalibration:
    """Pool-adjacent-violators monotone calibration on a rolling window.

    Fits the isotonic regression of measured acceptance on predicted
    acceptance; `calibrate(p)` returns the monotone-adjusted prediction so the
    router's expected-acceptance order is preserved while the level tracks
    reality. Synthetic-skewed data keeps the fit monotone (acceptance test).
    """

    def __init__(self, window: int = 256) -> None:
        self.window = window
        self._pairs: list[tuple[float, float]] = []
        self._fit: dict[float, float] = {}

    def update(self, predicted: float, measured: float) -> None:
        self._pairs.append((float(predicted), float(measured)))
        if len(self._pairs) > self.window:
            self._pairs = self._pairs[-self.window:]
        self._refit()

    def _refit(self) -> None:
        pts = sorted(set(self._pairs), key=lambda x: x[0])
        if not pts:
            self._fit = {}
            return
        # PAV: pool adjacent violators until monotone non-decreasing means.
        blocks: list[list[tuple[float, float]]] = [[pts[0]]]
        for p in pts[1:]:
            blocks.append([p])
            while len(blocks) >= 2:
                a = sum(y for _, y in blocks[-2]) / len(blocks[-2])
                b = sum(y for _, y in blocks[-1]) / len(blocks[-1])
                if a <= b:
                    break
                merged = blocks[-2] + blocks[-1]
                blocks = blocks[:-2] + [merged]
        self._fit = {}
        for blk in blocks:
            level = sum(y for _, y in blk) / len(blk)
            for x, _ in blk:
                self._fit[x] = level

    def calibrate(self, p: float) -> float:
        if not self._fit:
            return float(p)
        keys = sorted(self._fit)
        # monotone piecewise-constant interpolation
        prev = self._fit[keys[0]]
        for k in keys:
            if p <= k:
                return prev
            prev = self._fit[k]
        return prev

    @property
    def is_fitted(self) -> bool:
        return bool(self._fit)


def abstention_roc(scores: list[tuple[float, int]]) -> dict:
    """Precision/recall + AUC over (predicted utility, positive-outcome) pairs.

    positive = speculation beat the vanilla baseline (net positive speedup).
    Returns ROC data with a threshold sweep and AUC (trapezoid rule).
    """
    if not scores:
        return {"auc": 0.0, "thresholds": [], "fpr": [], "tpr": []}
    thresholds = sorted({s for s, _ in scores}, reverse=True)
    pos = sum(y for _, y in scores)
    neg = len(scores) - pos
    fprs: list[float] = []
    tprs: list[float] = []
    for t in thresholds:
        tp = sum(y for s, y in scores if s >= t)
        fp = sum(1 for s, y in scores if s >= t and y == 0)
        tprs.append(tp / max(1, pos))
        fprs.append(fp / max(1, neg))
    # AUC by trapezoid over (fpr, tpr), descending threshold order.
    auc = 0.0
    for i in range(1, len(fprs)):
        x0, x1 = fprs[i - 1], fprs[i]
        y0, y1 = tprs[i - 1], tprs[i]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return {"auc": round(auc, 4), "thresholds": thresholds, "fpr": fprs, "tpr": tprs}
