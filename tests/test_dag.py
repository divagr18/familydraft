"""Golden tests for the candidate trie/DAG builder (plan todo 7).

Goldens mirror the worked examples in the concept note: the two-expert
prefix merge (D:\MoE\family_draft_moe_concept.md lines 143-152) and the
rejection-memory branching example (lines 282-290).
"""

from __future__ import annotations

from familydraft.verify.dag import CandidateDag


def test_concept_note_merge_golden() -> None:
    # Expert A proposes tokens [1, 2, 3, 4]  ("return result\n}")
    # Expert B proposes tokens [1, 2, 3, 5, 4]  ("return result\n\n}")
    dag = CandidateDag()
    dag.insert([1, 2, 3, 4], expert_id=0, confidence=0.9, router_prob=0.6)
    dag.insert([1, 2, 3, 5, 4], expert_id=1, confidence=0.8, router_prob=0.4)

    # root + 3 shared prefix + 2 divergent + 1 re-converged token = 7
    assert dag.node_count == 7

    prefix = dag.get([1, 2, 3])
    assert prefix is not None
    node = dag.node(prefix)
    assert node.support == 2
    assert node.experts == frozenset({0, 1})
    assert node.confidence == 0.9
    assert node.router_prob == 0.6
    assert node.depth == 3

    branch_tokens = sorted(dag.children(prefix).keys())
    assert branch_tokens == [4, 5]
    branch_a = dag.children(prefix)[4]
    branch_b = dag.children(prefix)[5]
    assert dag.children(branch_a) == {}
    assert sorted(dag.children(branch_b).keys()) == [4]
    assert dag.node(branch_b).support == 1
    assert dag.node(dag.children(branch_b)[4]).support == 1


def test_concept_note_rejection_memory_golden() -> None:
    # Proposals "\n-" ([9, 20]) and repaired "\n\n-" ([9, 9, 20])
    dag = CandidateDag()
    dag.insert([9, 20], expert_id=0, confidence=0.7)
    dag.insert([9, 9, 20], expert_id=1, confidence=0.5)

    # root + "\n" + {"-", "\n"} + terminal "-" = 5
    assert dag.node_count == 5
    first = dag.get([9])
    assert dag.node(first).support == 2
    assert sorted(dag.children(first).keys()) == [9, 20]
    repaired_mid = dag.children(first)[9]
    assert sorted(dag.children(repaired_mid).keys()) == [20]
    assert dag.node(dag.children(repaired_mid)[20]).support == 1


def test_reinsert_same_expert_does_not_double_support() -> None:
    dag = CandidateDag()
    dag.insert([7, 8], expert_id=0, confidence=0.4)
    dag.insert([7, 8, 9], expert_id=0, confidence=0.6)
    shared = dag.get([7, 8])
    node = dag.node(shared)
    assert node.support == 1
    assert node.experts == frozenset({0})
    assert node.confidence == 0.6


def test_budget_prune_is_deterministic_and_ancestor_closed() -> None:
    def build() -> CandidateDag:
        d = CandidateDag()
        for expert in range(8):
            tokens = [expert * 10 + i for i in range(1, 6)]
            d.insert(tokens, expert_id=expert)
        return d

    full = build()
    assert full.node_count == 41

    pruned = build()
    pruned.prune_to_budget(16)
    assert pruned.node_count == 16

    # Determinism: identical rebuild + prune yields identical survivors
    again = build()
    again.prune_to_budget(16)
    assert [n.token_id for n in pruned.nodes_topo()] == [n.token_id for n in again.nodes_topo()]

    # Ancestor closure + root survival
    ids = {n.node_id for n in pruned.nodes_topo()}
    assert 0 in ids
    for n in pruned.nodes_topo():
        if n.parent_id is not None:
            assert n.parent_id in ids

    # With 8 disjoint equal-support 5-token chains, trimming to 16 removes
    # the five highest-expert chains entirely, leaving experts 0, 1, 2
    surviving_tokens = {n.token_id for n in pruned.nodes_topo() if n.token_id is not None}
    expected = {e * 10 + i for e in range(3) for i in range(1, 6)}
    assert surviving_tokens == expected


def test_budget_prune_noop_when_within_budget() -> None:
    dag = CandidateDag()
    dag.insert([1, 2], expert_id=0)
    dag.prune_to_budget(16)
    assert dag.node_count == 3


def test_topological_export_orders_parents_first() -> None:
    dag = CandidateDag()
    dag.insert([1, 2, 3], expert_id=0)
    dag.insert([1, 4], expert_id=1)
    order = dag.nodes_topo()
    position = {n.node_id: i for i, n in enumerate(order)}
    assert order[0].node_id == 0 and order[0].parent_id is None
    for n in order[1:]:
        assert position[n.parent_id] < position[n.node_id]


def test_empty_dag_and_lookup_misses() -> None:
    dag = CandidateDag()
    assert dag.node_count == 1
    assert dag.get([1]) is None
    dag.prune_to_budget(1)
    assert dag.node_count == 1
