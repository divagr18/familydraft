"""Candidate trie/DAG builder for speculative proposals (plan todo 7).

Each expert's token proposal is inserted into one shared trie so agreement
survives as a long shared prefix and only actual disagreement creates extra
verification nodes. Structure and bookkeeping only — scoring policies live
in the router (todo 19), verification in the DAG verifier (todo 8).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DagNode:
    """Immutable view of one trie node."""

    node_id: int
    token_id: int | None
    parent_id: int | None
    depth: int
    experts: frozenset[int]
    confidence: float
    router_prob: float
    cost: float

    @property
    def support(self) -> int:
        return len(self.experts)


class CandidateDag:
    """Token trie over family-vocab proposals with per-node metadata.

    Node ids are stable for the DAG's lifetime. Budget pruning uses lazy
    deletion, so a surviving node's id never changes.
    """

    def __init__(self) -> None:
        self._tokens: list[int | None] = [None]
        self._parents: list[int | None] = [None]
        self._depths: list[int] = [0]
        self._experts: list[set[int]] = [set()]
        self._confidence: list[float] = [0.0]
        self._router_prob: list[float] = [0.0]
        self._cost: list[float] = [0.0]
        self._children: list[dict[int, int]] = [{}]
        self._dead: set[int] = set()

    @property
    def node_count(self) -> int:
        return len(self._tokens) - len(self._dead)

    def insert(
        self,
        tokens: Sequence[int],
        expert_id: int,
        confidence: float = 0.0,
        router_prob: float = 0.0,
        cost: float = 0.0,
    ) -> list[int]:
        """Insert one proposal; existing prefixes gain support, not nodes."""
        if not tokens:
            raise ValueError("proposal must be non-empty")
        path = [0]
        cur = 0
        for depth, tok in enumerate(tokens, start=1):
            tok = int(tok)
            nxt = self._children[cur].get(tok)
            if nxt is None or nxt in self._dead:
                nxt = self._new_node(tok, cur, depth)
                self._children[cur][tok] = nxt
            cur = nxt
            self._experts[cur].add(expert_id)
            self._confidence[cur] = max(self._confidence[cur], confidence)
            self._router_prob[cur] = max(self._router_prob[cur], router_prob)
            self._cost[cur] = max(self._cost[cur], cost)
            path.append(cur)
        return path

    def get(self, path: Sequence[int]) -> int | None:
        cur = 0
        for tok in path:
            nxt = self._children[cur].get(int(tok))
            if nxt is None or nxt in self._dead:
                return None
            cur = nxt
        return cur

    def node(self, node_id: int) -> DagNode:
        if node_id in self._dead:
            raise KeyError(f"node {node_id} was pruned")
        return DagNode(
            node_id=node_id,
            token_id=self._tokens[node_id],
            parent_id=self._parents[node_id],
            depth=self._depths[node_id],
            experts=frozenset(self._experts[node_id]),
            confidence=self._confidence[node_id],
            router_prob=self._router_prob[node_id],
            cost=self._cost[node_id],
        )

    def children(self, node_id: int) -> dict[int, int]:
        return {
            tok: child
            for tok, child in self._children[node_id].items()
            if child not in self._dead
        }

    def nodes_topo(self) -> list[DagNode]:
        """BFS order: every parent precedes its children."""
        out: list[DagNode] = []
        queue = deque([0])
        while queue:
            nid = queue.popleft()
            if nid in self._dead:
                continue
            out.append(self.node(nid))
            queue.extend(sorted(self.children(nid).values()))
        return out

    def branches(self) -> list[tuple[int, ...]]:
        """Maximal root-to-leaf paths in deterministic order.

        DFS with children visited in ascending token order, so seeded
        per-branch processing (e.g. RNG tapes) is reproducible.
        """
        out: list[tuple[int, ...]] = []

        def walk(node: int, prefix: list[int]) -> None:
            kids = self.children(node)
            if not kids and node != 0:
                out.append(tuple(prefix))
                return
            for tok in sorted(kids):
                walk(kids[tok], prefix + [tok])

        walk(0, [])
        return out

    def prune_to_budget(self, max_nodes: int) -> None:
        """Trim to max_nodes by repeatedly removing the weakest leaf.

        Leaf order is (support, confidence) ascending with ties broken by
        largest node id, so the outcome is deterministic and ancestor
        closure holds by construction (only leaves are removed).
        """
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1 (the root always stays)")
        while self.node_count > max_nodes:
            live_children = [self.children(i) for i in range(len(self._tokens))]
            leaves = [
                i
                for i in range(len(self._tokens))
                if i not in self._dead and i != 0 and not live_children[i]
            ]
            if not leaves:
                break
            victim = min(
                leaves,
                key=lambda i: (len(self._experts[i]), self._confidence[i], -i),
            )
            self._dead.add(victim)

    def _new_node(self, token_id: int, parent_id: int, depth: int) -> int:
        self._tokens.append(token_id)
        self._parents.append(parent_id)
        self._depths.append(depth)
        self._experts.append(set())
        self._confidence.append(0.0)
        self._router_prob.append(0.0)
        self._cost.append(0.0)
        self._children.append({})
        return len(self._tokens) - 1
