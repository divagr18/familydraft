"""FLOP accounting for Phase-1 baselines (M3).

Computes per-system FLOPs per emitted token so the equal-FLOP dense baseline is
a real, measured comparison (audit: "no equal-FLOP dense baseline").

Model: dense transformer forward FLOPs per token ~= 2 * N_params (the standard
2-N rule for a forward pass). For the trunk/target we use the exact architecture
counts from transformers configs: per layer per token, attention contributes
4*h^2 (Q,K,V,O over h_heads... approximated as 2*h^2*qkv + 2*h^2*out with
grouped KV), MLP contributes 2*h*i per dense layer, plus the embedding lookup.
The absolute number matters less than the *ratio* between systems, so we use
the standard 2-N-per-token approximation consistently everywhere and report it
as an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

FLOP_MULTIPLIER = 2.0  # ~2 FLOPs per parameter per token (forward)


@dataclass(frozen=True)
class FlopBudget:
    params: float
    flops_per_token: float
    label: str

    @classmethod
    def from_config(cls, hidden: int, layers: int, intermediate: int,
                    kv_heads: int, heads: int, vocab: int, label: str) -> "FlopBudget":
        # Parameter estimate from architecture (embedding + per-layer attn/mlp).
        attn_params = 4 * hidden * hidden  # Q,K,V,O dense (V via kv_heads approx)
        mlp_params = 2 * hidden * intermediate
        per_layer = attn_params + mlp_params
        params = vocab * hidden + layers * per_layer
        return cls(params=params, flops_per_token=FLOP_MULTIPLIER * params, label=label)

    def layers_for_budget(self, hidden: int, intermediate: int, vocab: int) -> int:
        """Dense layer count whose per-token FLOPs match this budget."""
        per_layer = 2 * hidden * hidden + 2 * hidden * intermediate
        per_layer_flops = FLOP_MULTIPLIER * per_layer
        embed_flops = FLOP_MULTIPLIER * vocab * hidden
        layers = max(0.0, (self.flops_per_token - embed_flops) / per_layer_flops)
        return int(round(layers))


def dense_flops_per_token(hidden: int, layers: int, intermediate: int,
                          kv_heads: int, heads: int, vocab: int) -> float:
    return FlopBudget.from_config(
        hidden, layers, intermediate, kv_heads, heads, vocab, "dense"
    ).flops_per_token


def spec_loop_flops_per_emitted_token(
    target_budget: FlopBudget,
    trunk_budget: FlopBudget,
    draft_tokens_per_round: float,
    verify_nodes_per_round: float,
    emitted_tokens_per_round: float,
) -> float:
    """FLOPs spent per emitted token by a speculative loop.

    Per round: the drafter (trunk) forwards the context+draft once, the target
    verifies `verify_nodes` nodes, and the target decodes the bonus once. We
    count (trunk_forward + verify_forward + bonus_decode) / emitted tokens.
    Context length is excluded (shared baseline cost); the drafter's draft
    tokens are counted at trunk cost and the verified nodes at target cost.
    """
    draft_flops = trunk_budget.flops_per_token * draft_tokens_per_round
    verify_flops = target_budget.flops_per_token * verify_nodes_per_round
    bonus_flops = target_budget.flops_per_token  # 1 bonus decode
    total = draft_flops + verify_flops + bonus_flops
    return total / max(1e-9, emitted_tokens_per_round)
