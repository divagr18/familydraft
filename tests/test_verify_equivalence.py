"""M1 equivalence gate: DAG verifier == reference sequential verifier.

Cases use synthetic hash-seeded distribution oracles (never the
implementation under test) plus one real-model case on Qwen3-0.6B.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from familydraft.verify.dag import CandidateDag
from familydraft.verify.dag_verifier import verify_dag_greedy, verify_dag_sample
from familydraft.verify.reference import verify_chain_greedy, verify_chain_sample

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)

GREEDY_CASES = 500
SAMPLE_CASES = 200
_TIE_BREAK_STRIDE = 1_000_003


def _fnv1a(prefix: tuple[int, ...]) -> int:
    h = 0x84222325CBF29CE7
    for tok in prefix:
        h = ((h ^ int(tok)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def make_dist_oracle(base_seed: int, vocab: int):
    """Pure deterministic oracle: prefix -> normalized distribution."""

    def at(prefix: tuple[int, ...]) -> torch.Tensor:
        seed = (base_seed * 0x9E3779B1 ^ _fnv1a(prefix)) & 0x7FFFFFFF
        gen = torch.Generator().manual_seed(seed)
        raw = torch.rand(vocab, generator=gen) + 1e-3
        return raw / raw.sum()

    return at


def build_random_dag(gen: random.Random, vocab: int) -> CandidateDag:
    n_branches = gen.randint(1, 3)
    base = [gen.randrange(vocab) for _ in range(gen.randint(1, 8))]
    dag = CandidateDag()
    for b in range(n_branches):
        split = gen.randint(0, len(base)) if gen.random() < 0.5 else gen.randint(1, len(base))
        keep = base[:split]
        tail_len = gen.randint(1, max(1, 8 - len(keep)))
        proposal = keep + [gen.randrange(vocab) for _ in range(tail_len)]
        dag.insert(proposal, expert_id=b)
    return dag


def test_greedy_dag_verifier_matches_reference_on_500_random_cases() -> None:
    vocab = 12
    for case in range(GREEDY_CASES):
        gen = random.Random(case)
        dag = build_random_dag(gen, vocab)
        oracle = make_dist_oracle(base_seed=case, vocab=vocab)
        calls = 0

        def counting(prefix: tuple[int, ...]) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return oracle(prefix)

        got = {o.branch: o.verdict for o in verify_dag_greedy(dag, counting)}
        unique_prefixes = {()}
        for branch in dag.branches():
            targets = [oracle(branch[:k]) for k in range(len(branch) + 1)]
            ref = verify_chain_greedy(targets, list(branch))
            assert got[branch] == ref, f"case={case} branch={branch}"
            for k in range(len(branch) + 1):
                unique_prefixes.add(branch[:k])
        assert calls <= len(unique_prefixes), f"case={case}: recomputed distributions"


def test_sampling_dag_verifier_matches_reference_on_seeded_tapes() -> None:
    vocab = 10
    for case in range(SAMPLE_CASES):
        gen = random.Random(case ^ 0xA5A5)
        dag = build_random_dag(gen, vocab)
        target_at = make_dist_oracle(base_seed=case * 7 + 1, vocab=vocab)
        draft_at = make_dist_oracle(base_seed=case * 7 + 3, vocab=vocab)
        got = {
            o.branch: o.verdict
            for o in verify_dag_sample(dag, target_at, draft_at, seed=case)
        }
        for i, branch in enumerate(dag.branches()):
            targets = [target_at(branch[:k]) for k in range(len(branch) + 1)]
            drafts = [draft_at(branch[:k]) for k in range(len(branch))]
            ref_gen = torch.Generator().manual_seed(case + _TIE_BREAK_STRIDE * i)
            ref = verify_chain_sample(targets, drafts, list(branch), generator=ref_gen)
            assert got[branch] == ref, f"case={case} branch={branch}"


@pytest.mark.gpu
@requires_cuda
def test_real_model_greedy_equivalence_on_qwen3_06b() -> None:
    from functools import lru_cache

    from familydraft.targets.wrapper import TargetModel

    target = TargetModel.load("Qwen/Qwen3-0.6B", dtype="bf16")
    golden = json.loads(
        (Path(__file__).parent / "goldens" / "target_wrapper_greedy.txt").read_text(
            encoding="utf-8"
        )
    )
    prompt_ids = target.tokenizer(golden["prompt"], return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    vocab = target.vocab_size
    cont = golden["new_token_ids"]
    branch_long = tuple(cont[:6])
    branch_mid = tuple(cont[:4]) + ((cont[4] + 1) % vocab,)
    branch_short = tuple(cont[:2]) + ((cont[2] + 1) % vocab,)

    dag = CandidateDag()
    for b, branch in enumerate((branch_long, branch_mid, branch_short)):
        dag.insert(list(branch), expert_id=b)
    assert dag.node_count == 9  # 8 trie nodes + virtual root

    @lru_cache(maxsize=None)
    def dist_at(prefix: tuple[int, ...]) -> torch.Tensor:
        ids = torch.cat([prompt_ids, torch.tensor(prefix, dtype=torch.long)])
        snap = target.topk_logits(ids.unsqueeze(0), k=1)
        argmax_token = int(snap.token_ids[-1, 0])
        onehot = torch.zeros(vocab, dtype=torch.float32)
        onehot[argmax_token] = 1.0
        return onehot

    got = {o.branch: o.verdict for o in verify_dag_greedy(dag, dist_at)}
    for branch in (branch_long, branch_mid, branch_short):
        targets = [dist_at(branch[:k]) for k in range(len(branch) + 1)]
        ref = verify_chain_greedy(targets, list(branch))
        assert got[branch] == ref, f"branch={branch}"
