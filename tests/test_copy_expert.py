"""Copy/retrieval expert tests (plan todo 17)."""

from __future__ import annotations

import time

from familydraft.experts.copy import CopyExpert, CopyProposal


def _tok_class(t: int) -> str:
    if 100 <= t < 200:
        return "number"
    if 500 <= t < 600:
        return "identifier"
    return "other"


def test_golden_repeated_block_copies_following_span() -> None:
    block = [10, 11, 12, 13]
    following = [20, 21]
    tokens = block + following + [99] + block
    expert = CopyExpert(seed=4, min_length=3)
    expert.build_index(tokens)
    proposal = expert.propose(tokens, max_new=2)
    assert proposal is not None
    assert proposal.candidate_ids == (20, 21)
    assert proposal.match_length == 4


def test_golden_json_repetition() -> None:
    json_block = [300, 301, 302, 303, 304, 305]
    sep = [1, 2]
    tokens = json_block + sep + [400, 401] + json_block
    expert = CopyExpert(seed=4, min_length=3)
    expert.build_index(tokens)
    proposal = expert.propose(tokens, max_new=4)
    assert proposal is not None
    assert proposal.candidate_ids == (1, 2, 400, 401)


def test_slot_filling_substitutes_from_context() -> None:
    context = [501, 101, 50]
    candidate = [502, 102, 55]
    filled = CopyExpert.fill_slots(candidate, context, _tok_class)
    assert filled == [501, 101, 55]


def test_slot_filling_abstains_when_class_absent() -> None:
    context = [50, 51]
    candidate = [503]
    assert CopyExpert.fill_slots(candidate, context, _tok_class) is None


def test_abstains_when_no_match_meets_min_length() -> None:
    tokens = list(range(0, 100))
    expert = CopyExpert(seed=4, min_length=3)
    expert.build_index(tokens)
    context = [97, 98, 99, 1000, 1001, 1002, 1003]
    assert expert.propose(context, max_new=4) is None


def test_empty_index_returns_none() -> None:
    expert = CopyExpert(seed=4)
    expert.build_index([])
    assert expert.propose([1, 2, 3, 4, 5]) is None


def test_context_longer_than_index_is_graceful() -> None:
    expert = CopyExpert(seed=4)
    expert.build_index([1, 2, 3, 4])
    long_context = [1, 2, 3, 4] + list(range(100, 130))
    result = expert.propose(long_context, max_new=4)
    assert result is None or isinstance(result, CopyProposal)


def test_query_latency_within_budget() -> None:
    import random

    rng = random.Random(0)
    tokens = [rng.randrange(0, 5000) for _ in range(8000)]
    expert = CopyExpert(seed=4, min_length=3, max_index=8000)
    expert.build_index(tokens)
    latencies = []
    for i in range(1000):
        start = (i * 7) % (len(tokens) - 16)
        context = tokens[start : start + 12]
        t0 = time.perf_counter()
        expert.propose(context, max_new=8)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    p50_ms = latencies[len(latencies) // 2] * 1000
    assert p50_ms < 2.0, f"p50 query latency {p50_ms:.3f}ms exceeds 2ms budget"
