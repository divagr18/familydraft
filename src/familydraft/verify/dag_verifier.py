"""DAG candidate verifier, v1 (memoized prefix forwards).

Verifies every branch of a candidate DAG against a target distribution
oracle. Each node's distribution is computed exactly once and shared
across branches — the computation pattern of a single joint forward,
expressed without kernel batching. Per-branch acceptance is expected to
be bit-identical to the reference sequential verifier; that equivalence
is the M1 gate, proven in tests/test_verify_equivalence.py. Acceptance
math is deliberately re-implemented here rather than delegated, so the
gate compares two independent code paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .dag import CandidateDag
from .reference import ChainVerdict

DistAt = Callable[[tuple[int, ...]], torch.Tensor]

_RESIDUAL_EPS = 1e-3


@dataclass(frozen=True)
class BranchOutcome:
    branch: tuple[int, ...]
    verdict: ChainVerdict


def _memoized(dist_at: DistAt) -> tuple[DistAt, Callable[[], int]]:
    memo: dict[tuple[int, ...], torch.Tensor] = {}

    def lookup(prefix: tuple[int, ...]) -> torch.Tensor:
        cached = memo.get(prefix)
        if cached is None:
            cached = dist_at(prefix)
            memo[prefix] = cached
        return cached

    return lookup, lambda: len(memo)


def verify_dag_greedy(dag: CandidateDag, dist_at: DistAt) -> list[BranchOutcome]:
    dist, _ = _memoized(dist_at)
    outcomes: list[BranchOutcome] = []
    for branch in dag.branches():
        accepted_len = 0
        for k, tok in enumerate(branch):
            if int(tok) == int(torch.argmax(dist(branch[:k]))):
                accepted_len += 1
            else:
                break
        bonus = int(torch.argmax(dist(branch[:accepted_len])))
        outcomes.append(
            BranchOutcome(
                branch=branch,
                verdict=ChainVerdict(tuple(branch[:accepted_len]), bonus),
            )
        )
    return outcomes


def verify_dag_sample(
    dag: CandidateDag,
    target_at: DistAt,
    draft_at: DistAt,
    seed: int,
) -> list[BranchOutcome]:
    target, _ = _memoized(target_at)
    draft, _ = _memoized(draft_at)
    outcomes: list[BranchOutcome] = []
    for i, branch in enumerate(dag.branches()):
        gen = torch.Generator().manual_seed(seed + 1_000_003 * i)
        accepted_len = 0
        bonus: int | None = None
        for k, tok in enumerate(branch):
            tok = int(tok)
            p = target(branch[:k])
            q = draft(branch[:k])
            ratio = (p[tok] / q[tok]) if q[tok] > 0 else torch.tensor(0.0, dtype=p.dtype)
            if torch.rand((), generator=gen) < ratio:
                accepted_len += 1
                continue
            residual = torch.clamp(p - q, min=0)
            if residual.sum() <= _RESIDUAL_EPS:
                residual = p.clone()
            bonus = int(torch.multinomial(residual / residual.sum(), 1, generator=gen))
            break
        if bonus is None:
            bonus = int(torch.multinomial(target(branch), 1, generator=gen))
        outcomes.append(
            BranchOutcome(
                branch=branch,
                verdict=ChainVerdict(tuple(branch[:accepted_len]), bonus),
            )
        )
    return outcomes
