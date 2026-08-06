"""Reference sequential speculative-acceptance verifier.

Ground truth for the whole project: every faster verifier (DAG, tree) is
checked against these functions, so they favor clarity over speed. They
operate on explicit per-position distributions, never on model internals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

_PROB_SUM_ATOL = 1e-3


@dataclass(frozen=True)
class ChainVerdict:
    """Outcome of verifying one draft chain.

    accepted_tokens: draft tokens the target confirmed, in order.
    bonus_token: the target's own token emitted after the accepted prefix
    (the replacement token at a rejection, or the continuation after a full
    accept). Always present.
    """

    accepted_tokens: tuple[int, ...]
    bonus_token: int

    @property
    def accept_length(self) -> int:
        return len(self.accepted_tokens)


def verify_chain_greedy(
    target_dists: Sequence[torch.Tensor],
    draft_tokens: Sequence[int],
) -> ChainVerdict:
    """Greedy acceptance: longest prefix where draft == target argmax.

    target_dists[t] must be the target's distribution over the token that
    follows prefix + draft[:t]; logits or probabilities both work (argmax is
    invariant to softmax). Requires len(target_dists) >= len(draft) + 1 so
    the bonus position is defined.
    """
    if len(target_dists) < len(draft_tokens) + 1:
        raise ValueError(
            f"need at least {len(draft_tokens) + 1} target distributions "
            f"(draft length + bonus position), got {len(target_dists)}"
        )
    accepted: list[int] = []
    for t, d in enumerate(draft_tokens):
        if int(d) == int(torch.argmax(target_dists[t])):
            accepted.append(int(d))
        else:
            break
    return ChainVerdict(
        accepted_tokens=tuple(accepted),
        bonus_token=int(torch.argmax(target_dists[len(accepted)])),
    )


def _check_normalized(probs: torch.Tensor, where: str) -> None:
    if probs.ndim != 1 or probs.min() < 0 or not torch.allclose(
        probs.sum(), torch.tensor(1.0, dtype=probs.dtype), atol=_PROB_SUM_ATOL
    ):
        raise ValueError(f"{where} must be a non-negative 1-D distribution summing to 1")


def verify_chain_sample(
    target_probs: Sequence[torch.Tensor],
    draft_probs: Sequence[torch.Tensor],
    draft_tokens: Sequence[int],
    generator: torch.Generator,
) -> ChainVerdict:
    """Standard speculative-sampling acceptance (Leviathan et al. 2023).

    At position t with drafted token x: accept with probability
    min(1, p_t(x)/q_t(x)); on rejection, the bonus is sampled from the
    adjusted residual norm(max(0, p_t - q_t)). On full acceptance the bonus
    is sampled from the target distribution at the final position. With a
    seeded generator the decision tape is reproducible. Requires
    len(target_probs) >= len(draft) + 1 and len(draft_probs) >= len(draft).
    """
    if len(target_probs) < len(draft_tokens) + 1:
        raise ValueError(
            f"need at least {len(draft_tokens) + 1} target distributions "
            f"(draft length + bonus position), got {len(target_probs)}"
        )
    if len(draft_probs) < len(draft_tokens):
        raise ValueError(
            f"need at least {len(draft_tokens)} draft distributions, "
            f"got {len(draft_probs)}"
        )
    for t, p in enumerate(target_probs):
        _check_normalized(p, f"target_probs[{t}]")
    for t, q in enumerate(draft_probs):
        _check_normalized(q, f"draft_probs[{t}]")

    accepted: list[int] = []
    for t, x in enumerate(draft_tokens):
        x = int(x)
        p, q = target_probs[t], draft_probs[t]
        ratio = (p[x] / q[x]) if q[x] > 0 else torch.tensor(0.0, dtype=p.dtype)
        if torch.rand((), generator=generator) < ratio:
            accepted.append(x)
            continue
        residual = torch.clamp(p - q, min=0)
        if residual.sum() <= _PROB_SUM_ATOL:
            residual = p.clone()
        bonus = int(torch.multinomial(residual / residual.sum(), 1, generator=generator))
        return ChainVerdict(accepted_tokens=tuple(accepted), bonus_token=bonus)
    bonus = int(
        torch.multinomial(target_probs[len(draft_tokens)], 1, generator=generator)
    )
    return ChainVerdict(accepted_tokens=tuple(accepted), bonus_token=bonus)
