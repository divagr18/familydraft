"""Utility router v1 (plan todo 19).

A contextual-bandit router that picks 0, 1, or 2 experts per decoding state.
It maximises expected accepted tokens per wall-clock cost:

    U(e) = E[A_e] / (C_draft_e + C_verify(marginal_nodes_e))

E[A_e] is a linear contextual bandit over features (trunk summary, parser /
repetition / copy scores, target one-hot). Cost comes from the todo-9 cost
curve. The second expert is chosen to maximise marginal utility, discounted by
historical overlap with the first. Abstains when max utility < tau_abstain.
Weights are initialised offline (cold start) and refined online - there is no
static offline routing at runtime (per the Not-a-Bandit finding), and no
neural router in v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HORIZONS = (2, 4, 6, 8)

# Feature layout: [trunk_summary, parser_score, repetition_score, copy_score,
#                  target_onehot(num_targets)]
SCALAR_FEATURES = ("trunk_summary", "parser_score", "repetition_score", "copy_score")


def make_features(
    trunk_summary: float,
    parser_score: float,
    repetition_score: float,
    copy_score: float,
    target_id: int,
    num_targets: int = 7,
) -> list[float]:
    onehot = [0.0] * num_targets
    if 0 <= target_id < num_targets:
        onehot[target_id] = 1.0
    return [trunk_summary, parser_score, repetition_score, copy_score] + onehot


@dataclass
class RouterDecision:
    expert_subset: list[str]
    horizons: dict[str, int]
    abstain: bool
    utilities: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertStats:
    accepted_len_ema: float = 1.0
    first_rejection_ema: float = 1.0
    draft_ms_ema: float = 0.0
    overlap: dict[str, float] = field(default_factory=dict)


class UtilityRouter:
    def __init__(
        self,
        expert_names: list[str],
        draft_ms: dict[str, float],
        verify_ms_by_nodes: dict[int, float],
        tau_abstain: float = 0.05,
        ema_rate: float = 0.2,
        feature_dim: int = 4,
        num_targets: int = 7,
        always_on_cost_ms: dict[str, float] | None = None,
        routing_mode: str = "utility",
    ) -> None:
        self.expert_names = list(expert_names)
        self.draft_ms = dict(draft_ms)
        self.verify_ms_per_node = self._marginal_verify_cost(verify_ms_by_nodes)
        self.tau_abstain = tau_abstain
        self.ema_rate = ema_rate
        self.feature_dim = feature_dim
        self.num_targets = num_targets
        # "utility" divides expected acceptance by cost; "acceptance" ignores
        # cost entirely (the acceptance-routing vs latency-utility ablation).
        if routing_mode not in ("utility", "acceptance"):
            raise ValueError(
                f"routing_mode must be 'utility' or 'acceptance', got {routing_mode!r}"
            )
        self.routing_mode = routing_mode
        # Experts whose drafting cost is paid every round regardless of
        # selection (e.g. copy is always-on): their draft cost is sunk, so
        # selection utility counts only the marginal verify cost.
        self.always_on_cost_ms = dict(always_on_cost_ms or {})
        self.base: dict[str, float] = {e: 1.0 for e in expert_names}
        self.weights: dict[str, list[float]] = {
            e: [0.0] * (feature_dim + num_targets) for e in expert_names
        }
        self.stats: dict[str, ExpertStats] = {e: ExpertStats() for e in expert_names}

    @staticmethod
    def _marginal_verify_cost(curve: dict[int, float]) -> float:
        nodes = sorted(int(k) for k in curve)
        if len(nodes) < 2:
            return 1.0
        lo, hi = nodes[0], nodes[-1]
        span = hi - lo
        if span <= 0:
            return 1.0
        return (float(curve[hi]) - float(curve[lo])) / span

    def expected_acceptance(self, expert: str, features: list[float]) -> float:
        w = self.weights[expert]
        dot = sum(wi * xi for wi, xi in zip(w, features))
        raw = max(0.0, self.base[expert] + dot)
        cal = getattr(self, "calibrators", {}).get(expert)
        if cal is not None and cal.is_fitted:
            return max(0.0, cal.calibrate(raw))
        return raw

    def set_calibrators(self, calibrators: dict) -> None:
        """Attach per-expert isotonic calibrators (plan todo 26). The router's
        utility estimates then pass through the calibrated acceptance."""
        self.calibrators = dict(calibrators)

    def set_weights(self, weights: dict[str, list[float]]) -> None:
        """Load learned bandit weights (from rollout training)."""
        for expert, w in weights.items():
            if expert in self.weights and len(w) == len(self.weights[expert]):
                self.weights[expert] = [float(x) for x in w]

    def weights_json(self) -> dict[str, list[float]]:
        return {e: list(w) for e, w in self.weights.items()}

    def save_weights(self, path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.weights_json()), encoding="utf-8")

    @classmethod
    def load_weights(cls, path) -> dict[str, list[float]]:
        import json

        return json.loads(Path(path).read_text(encoding="utf-8"))

    def horizon_for(self, expert: str) -> int:
        acc = self.stats[expert].accepted_len_ema
        for h in HORIZONS:
            if acc <= h:
                return h
        return HORIZONS[-1]

    def utility(self, expert: str, features: list[float], horizon: int) -> float:
        if self.routing_mode == "acceptance":
            return self.expected_acceptance(expert, features)
        draft = 0.0 if expert in self.always_on_cost_ms else self.draft_ms.get(expert, 0.0)
        cost = draft + self.verify_ms_per_node * horizon
        if cost <= 0:
            return 0.0
        return self.expected_acceptance(expert, features) / cost

    def pair_overlap(self, a: str, b: str) -> float:
        return self.stats[a].overlap.get(b, 0.0)

    def select(self, features: list[float], max_experts: int = 2) -> RouterDecision:
        utils = {
            e: self.utility(e, features, self.horizon_for(e)) for e in self.expert_names
        }
        best = max(self.expert_names, key=lambda e: utils[e])
        if utils[best] < self.tau_abstain:
            return RouterDecision([], {}, True, utils)
        chosen = [best]
        horizons = {best: self.horizon_for(best)}
        if max_experts >= 2:
            best_marginal = 0.0
            second = None
            for cand in self.expert_names:
                if cand == best:
                    continue
                overlap = self.pair_overlap(best, cand)
                marginal = utils[cand] * (1.0 - overlap)
                if marginal > best_marginal:
                    best_marginal = marginal
                    second = cand
            if second is not None and best_marginal > 0:
                chosen.append(second)
                horizons[second] = self.horizon_for(second)
        return RouterDecision(chosen, horizons, False, utils)

    def update_feedback(
        self,
        expert: str,
        accepted_len: float,
        draft_ms: float,
        first_rejection: float,
    ) -> None:
        s = self.stats[expert]
        r = self.ema_rate
        s.accepted_len_ema = (1 - r) * s.accepted_len_ema + r * accepted_len
        s.draft_ms_ema = (1 - r) * s.draft_ms_ema + r * draft_ms
        s.first_rejection_ema = (1 - r) * s.first_rejection_ema + r * first_rejection
        self.draft_ms[expert] = s.draft_ms_ema
        # Adapt the expected-acceptance level toward realised acceptance so
        # selection tracks the live environment (the learned weights model the
        # feature response; base tracks the realised level).
        self.base[expert] = (1 - r) * self.base[expert] + r * max(0.0, accepted_len)
        # Feed the per-expert isotonic calibrator with (predicted, measured).
        cal = getattr(self, "calibrators", {}).get(expert)
        if cal is not None:
            cal.update(s.accepted_len_ema, max(0.0, accepted_len))

    def record_overlap(self, a: str, b: str, overlap: float) -> None:
        self.stats[a].overlap[b] = overlap
        self.stats[b].overlap[a] = overlap

    def cold_start(self, acceptance_by_expert: dict[str, float]) -> None:
        """Offline initialisation from simulated acceptance (todo-10 replay).

        Sets the bandit base utility from measured per-expert acceptance so the
        router is functional before any live rollout. accepted_len_ema is seeded
        at 1.0 (minimal horizon) rather than the base value, so cold-start
        selection is driven by expected quality (base), not by an inflated
        horizon that would make high-base experts look expensive and get skipped.
        Runtime behaviour then adapts online from feedback.
        """
        for expert, acc in acceptance_by_expert.items():
            if expert in self.base:
                self.base[expert] = max(0.0, float(acc))
                self.stats[expert].accepted_len_ema = 1.0
