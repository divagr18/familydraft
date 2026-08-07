"""Router-driven multi-expert DAG speculative decoding - the thesis system.

The pitch, made real: a utility router activates a small subset of heterogeneous
experts per decoding state; each selected expert independently proposes a
continuation; proposals are merged into one candidate DAG (shared prefixes
stored once); the DAG is verified against the target; the best branch is
accepted and the rest discarded. Online feedback updates the router's per-expert
stats, and rejected branches feed the rejection-memory expert.

v1 verification verifies each DAG branch with a fresh copy of the KV cache
(deepcopy is cheap enough at eval scale and avoids cross-branch cache
contamination). Batched tree attention is a Phase-4 optimization, not needed for
correctness: every emitted token is the target's own argmax, so the output is
lossless by construction (proven end-to-end on fp32 numerics in tests).
"""

from __future__ import annotations

import copy

import torch

from familydraft.router.router import UtilityRouter, make_features


def make_macro_drafter(macro_expert, tokenizer):
    def draft_fn(context_ids: list[int]) -> list[int]:
        text = tokenizer.decode(context_ids, skip_special_tokens=True)
        proposals = macro_expert.propose_from_text(text, trunk_hidden=None, top_k=1)
        if not proposals:
            return []
        return list(proposals[0][1])

    return draft_fn


def make_reject_memory_drafter(memory, target_id, decode_mode: str = "greedy"):
    def draft_fn(context_ids: list[int]) -> list[int]:
        fingerprint = (tuple(context_ids[-8:]), "any", str(target_id), decode_mode)
        return list(memory.propose(fingerprint) or [])

    return draft_fn


class DagSpeculator:
    def __init__(
        self,
        target_model,
        router: UtilityRouter,
        experts: dict,
        horizons: dict,
        target_id: int = 0,
        memory=None,
        always_on: list[str] | None = None,
    ) -> None:
        self.model = target_model.model
        self.tokenizer = target_model.tokenizer
        self.router = router
        self.experts = dict(experts)
        self.horizons = dict(horizons)
        self.target_id = target_id
        self.memory = memory
        self.always_on = list(always_on or [])
        self.device = next(self.model.parameters()).device
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        self._eos = None if eos is None else int(eos)

    def _feed(self, past, tokens: list[int]):
        with torch.inference_mode():
            out = self.model(
                input_ids=torch.tensor([tokens], device=self.device),
                past_key_values=past,
                use_cache=True,
            )
        return out.past_key_values, out.logits[0]

    def _parser_score(self, context_ids: list[int]) -> float:
        from familydraft.experts.parse_state import parse_scan

        text = self.tokenizer.decode(context_ids, skip_special_tokens=True)
        state = parse_scan(text)
        return min(len(state.candidates) / 4.0, 1.0)

    def _features(self, context_ids: list[int], copy_score: float) -> list[float]:
        return make_features(
            0.0,
            self._parser_score(context_ids),
            copy_score,
            copy_score,
            self.target_id,
            self.router.num_targets,
        )

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> dict:
        x = torch.tensor([prompt_ids], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=x, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        ctx_len = len(prompt_ids)

        generated: list[int] = []
        rounds = 0
        accepted_tokens = 0

        while len(generated) < max_new_tokens:
            rounds += 1
            t_next = int(torch.argmax(logits, dim=-1))
            context_ids = prompt_ids + generated

            feats = self._features(context_ids, 0.0)
            decision = self.router.select(feats)
            # always_on experts (cheap structural ones) always propose; the
            # router only decides which ADDITIONAL experts to activate.
            selected = [
                e for e in self.always_on if e in self.experts
            ] + [e for e in decision.expert_subset if e in self.experts and e not in self.always_on]

            def emit_single() -> bool:
                nonlocal past, logits, ctx_len
                generated.append(t_next)
                if t_next == self._eos:
                    return True
                past, logits_next = self._feed(past, [t_next])
                logits = logits_next[-1]
                ctx_len += 1
                return False

            if decision.abstain or not selected:
                if emit_single():
                    break
                continue

            proposals: dict[str, list[int]] = {}
            for e in selected:
                horizon = self.horizons.get(e, 4)
                try:
                    prop = list(self.experts[e](context_ids))[:horizon]
                except Exception:
                    prop = []
                if prop:
                    proposals[e] = prop
            if not proposals:
                if emit_single():
                    break
                continue

            dag = None
            from familydraft.verify.dag import CandidateDag

            dag = CandidateDag()
            for eid, prop in enumerate(proposals.values()):
                dag.insert(prop, expert_id=eid)

            verified = self._verify_tree(past, ctx_len, t_next, dag, proposals, selected)
            if verified is None:
                if emit_single():
                    break
                continue
            best, per_expert_m, tree_cache = verified
            m, bonus, winner, accepted = best
            accepted_tokens += m

            final_cache = self._crop(tree_cache, ctx_len + m)
            past, logits_next = self._feed(final_cache, [bonus])
            logits = logits_next[-1]
            ctx_len += m + 1
            generated.extend(accepted)
            generated.append(bonus)
            if bonus == self._eos:
                break

            for e in selected:
                self.router.update_feedback(
                    e,
                    accepted_len=float(per_expert_m.get(e, 0)),
                    draft_ms=0.5,
                    first_rejection=float(per_expert_m.get(e, 0) + 1),
                )
            self._maybe_record_rejection(context_ids, winner, per_expert_m, proposals)

        return {
            "tokens": generated[:max_new_tokens],
            "rounds": rounds,
            "accepted_tokens": accepted_tokens,
            "tokens_per_round": len(generated) / max(1, rounds),
        }

    @staticmethod
    def _crop(cache, length: int):
        cache.crop(length)
        return cache

    @staticmethod
    def _branch_source(branch, proposals: dict, selected: list[str]) -> str:
        # proposals only contains experts that produced a non-empty proposal, so
        # iterate it (not `selected`, which may include experts that abstained).
        for e in proposals:
            if tuple(proposals[e]) == tuple(branch):
                return e
        return selected[0] if selected else ""

    def _verify_tree(self, base_cache, ctx_len, t_next, dag, proposals, selected):
        """Verify ALL DAG nodes in ONE forward (tree attention).

        Enumerates the DAG's non-root nodes in topological order, feeds them as a
        single sequence with a 0/-inf tree mask (node attends only to context +
        its ancestors) and depth-based position_ids, then walks each branch's
        acceptance from the per-node logits. Shared prefixes are computed once.
        Returns (best_branch, per_expert_m, tree_cache) where tree_cache holds
        context + all nodes and must be cropped by the caller.
        """
        NEG = float("-inf")
        tokens: list[int] = []
        parent: list[int] = []
        depth: list[int] = []
        row_by_id: dict[int, int] = {}
        for node in dag.nodes_topo():
            if node.node_id == 0:
                continue
            row_by_id[node.node_id] = len(tokens)
            tokens.append(int(node.token_id))
            parent.append(-1 if node.depth == 1 else row_by_id[node.parent_id])
            depth.append(node.depth)
        N = len(tokens)
        if N == 0:
            return None

        pos = [ctx_len + d - 1 for d in depth]
        mask = torch.full((1, 1, N, ctx_len + N), NEG, dtype=torch.bfloat16, device=self.device)
        mask[:, :, :, :ctx_len] = 0
        for i in range(N):
            mask[:, :, i, ctx_len + i] = 0
            p = parent[i]
            while p != -1:
                mask[:, :, i, ctx_len + p] = 0
                p = parent[p]

        tree_cache = copy.deepcopy(base_cache)
        with torch.inference_mode():
            out = self.model(
                input_ids=torch.tensor([tokens], device=self.device),
                past_key_values=tree_cache,
                attention_mask=mask,
                position_ids=torch.tensor([pos], device=self.device),
                use_cache=True,
            )
        lg = out.logits[0]
        tree_cache = out.past_key_values

        best = None
        per_expert_m: dict[str, int] = {}
        for branch in dag.branches():
            K = len(branch)
            if branch[0] != t_next:
                m = 0
                bonus = t_next
            else:
                m = 1
                bonus = t_next
                for j in range(1, K):
                    prev_node = dag.get(list(branch[:j]))
                    tt = int(torch.argmax(lg[row_by_id[prev_node]], dim=-1))
                    if branch[j] == tt:
                        m += 1
                    else:
                        bonus = tt
                        break
                else:
                    last_node = dag.get(list(branch[:K]))
                    bonus = int(torch.argmax(lg[row_by_id[last_node]], dim=-1))
            src = self._branch_source(branch, proposals, selected)
            per_expert_m[src] = max(per_expert_m.get(src, 0), m)
            if best is None or m > best[0]:
                best = (m, bonus, src, list(branch[:m]))
        return best, per_expert_m, tree_cache

    def _maybe_record_rejection(self, context_ids, winner, per_expert_m, proposals) -> None:
        if self.memory is None:
            return
        for e, m in per_expert_m.items():
            if m < len(proposals.get(e, [])):
                fingerprint = (tuple(context_ids[-8:]), "any", str(self.target_id), "greedy")
                try:
                    self.memory.record_rejection(fingerprint, replacement=[], accepted_suffix=[])
                except Exception:
                    pass


def build_dag_router(
    expert_names: list[str],
    draft_ms: dict[str, float],
    verify_ms_by_nodes: dict[int, float],
    base_acceptance: dict[str, float],
    tau_abstain: float = 0.05,
) -> UtilityRouter:
    router = UtilityRouter(
        expert_names=expert_names,
        draft_ms=draft_ms,
        verify_ms_by_nodes=verify_ms_by_nodes,
        tau_abstain=tau_abstain,
    )
    router.cold_start(base_acceptance)
    return router
