"""Tests for the router-driven multi-expert DAG speculator (thesis system).

The key invariant is the same as the chain loop: the speculative output must be
identical to vanilla greedy decoding under exact (fp32) numerics, no matter what
the experts propose. Verified on a tiny fp32 model so batch-vs-sequential bf16
artifacts cannot interfere.
"""

from __future__ import annotations

import types

import pytest
import torch

from familydraft.eval.draft_dag import (
    DagSpeculator,
    build_dag_router,
    make_macro_drafter,
)
from familydraft.eval.draft_loop import make_copy_drafter
from familydraft.experts.copy import CopyExpert
from familydraft.experts.macro import MacroExpert

CUDA = torch.cuda.is_available()

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Write a Python function that returns the sum of a list of numbers:"
REPETITIVE = "Write 8 list append statements adding i to a list for i from 0 to 7:"


def _wrap(model, tokenizer):
    return types.SimpleNamespace(model=model, tokenizer=tokenizer)


def _greedy(model, tokenizer, prompt: str, n: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(MODEL)
    pids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    with torch.inference_mode():
        out = model.generate(
            torch.tensor([pids], device=next(model.parameters()).device),
            max_new_tokens=n,
            do_sample=False,
        )
    return out[0, len(pids):].tolist()


def _make_router(expert_names, verify_curve):
    draft_ms = {e: 0.5 for e in expert_names}
    base = {e: 2.0 for e in expert_names}
    return build_dag_router(expert_names, draft_ms, verify_curve, base)


def _pids(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()


@pytest.fixture(scope="module")
def tiny_target():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(torch.float32)
    model.eval()
    if CUDA:
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return _wrap(model, tokenizer)


def _experts_for(tiny_target, names):
    from pathlib import Path

    from familydraft.experts.macro_render import build_renderer_from_config

    experts = {}
    if "copy" in names:
        ce = CopyExpert(seed=4, min_length=3)
        experts["copy"] = make_copy_drafter(ce, 4)
    if "macro" in names:
        tok = tiny_target.tokenizer
        renderer = build_renderer_from_config(Path(__file__).parent.parent, tok, tok.vocab_size)
        macro_expert = MacroExpert(renderer, head=None)
        experts["macro"] = make_macro_drafter(macro_expert, tok)
    return experts


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_single_expert_matches_vanilla(tiny_target) -> None:
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, PROMPT, 20)
    experts = _experts_for(tiny_target, ["copy"])
    router = _make_router(["copy"], {2: 40.0, 66: 1320.0})
    spec = DagSpeculator(tiny_target, router, experts, {"copy": 4})
    res = spec.generate(_pids(tiny_target.tokenizer, PROMPT), 20)
    assert res["tokens"] == vanilla


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_second_branch_winner_is_lossless(tiny_target) -> None:
    """Regression: when a NON-first branch wins, the continuation KV must be
    rebuilt from the context cache (a length-crop of the tree cache only works
    when the first branch wins). First expert proposes a 1-token match, second
    an oracle continuation -> second branch wins with m>=2."""
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, PROMPT, 20)
    pids = _pids(tiny_target.tokenizer, PROMPT)

    def first_draft(context_ids):
        # matches vanilla[0] only; diverges at position 1
        return [vanilla[0], (vanilla[1] + 1) % 151936, 5, 6]

    def oracle_draft(context_ids):
        # proposes the exact vanilla continuation -> wins
        offset = len(context_ids) - len(pids)
        return vanilla[offset: offset + 4]

    router = _make_router(["first", "second"], {2: 40.0, 66: 1320.0})
    spec = DagSpeculator(
        tiny_target,
        router,
        {"first": first_draft, "second": oracle_draft},
        {"first": 4, "second": 4},
        always_on=["first", "second"],
    )
    res = spec.generate(pids, 20)
    assert res["tokens"] == vanilla


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_second_branch_winner_real_model_deterministic(qwen_target) -> None:
    """Second-branch winner on the real model: deterministic across runs.
    (Byte-identity with the HF-sequential oracle is not asserted here: the
    tree verifier's logits come from a batched bf16 forward, so deep near-tie
    flips can reject the oracle - the documented batch-vs-sequential artifact.
    The tiny fp32 test is the exactness proof.)"""
    tok = qwen_target.tokenizer
    pids = _pids(tok, REPETITIVE)
    vanilla = _greedy(qwen_target.model, tok, REPETITIVE, 20)

    def first_draft(context_ids):
        return [vanilla[0], (vanilla[1] + 1) % 151643, 5, 6]

    def oracle_draft(context_ids):
        offset = len(context_ids) - len(pids)
        return vanilla[offset: offset + 4]

    router = _make_router(["first", "second"], {2: 40.0, 66: 1320.0})
    spec = DagSpeculator(
        qwen_target,
        router,
        {"first": first_draft, "second": oracle_draft},
        {"first": 4, "second": 4},
        always_on=["first", "second"],
    )
    r1 = spec.generate(pids, 20)
    r2 = spec.generate(pids, 20)
    assert r1["tokens"] == r2["tokens"]


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_rejection_memory_records_real_bonus(tiny_target) -> None:
    """A partially-rejected proposal must store the target's actual next token
    (per-branch bonus) as the correction, not an empty replacement."""
    from familydraft.experts.reject_memory import RejectionMemory

    pids = _pids(tiny_target.tokenizer, REPETITIVE)
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, REPETITIVE, 20)

    def half_wrong_draft(context_ids):
        return [vanilla[0], (vanilla[1] + 1) % 151936, 5, 6]

    memory = RejectionMemory(min_support=1)
    router = _make_router(["bad"], {2: 40.0, 66: 1320.0})
    spec = DagSpeculator(
        tiny_target,
        router,
        {"bad": half_wrong_draft},
        {"bad": 4},
        memory=memory,
        always_on=["bad"],
    )
    res = spec.generate(pids, 12)
    assert res["tokens"] == vanilla[:12]
    assert memory.size > 0, "no rejection recorded"
    entry = next(iter(memory._store.values()))
    assert len(entry.replacement) == 1, f"replacement should be the bonus, got {entry.replacement}"
    assert entry.replacement[0] == vanilla[1], (
        f"recorded replacement {entry.replacement[0]} != actual bonus {vanilla[1]}"
    )


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_multi_expert_matches_vanilla(tiny_target) -> None:
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, REPETITIVE, 20)
    experts = _experts_for(tiny_target, ["copy", "macro"])
    router = _make_router(["copy", "macro"], {2: 40.0, 66: 1320.0})
    spec = DagSpeculator(tiny_target, router, experts, {"copy": 4, "macro": 2})
    res = spec.generate(_pids(tiny_target.tokenizer, REPETITIVE), 20)
    assert res["tokens"] == vanilla


@pytest.fixture(scope="module")
def qwen_target():
    from familydraft.targets.wrapper import TargetModel

    return TargetModel.load(MODEL, dtype="bf16")


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_real_model_is_deterministic(qwen_target) -> None:
    from pathlib import Path

    from familydraft.experts.macro_render import build_renderer_from_config

    tok = qwen_target.tokenizer
    renderer = build_renderer_from_config(Path(__file__).parent.parent, tok, tok.vocab_size)
    macro_expert = MacroExpert(renderer, head=None)
    ce = CopyExpert(seed=4, min_length=3)
    experts = {
        "copy": make_copy_drafter(ce, 4),
        "macro": make_macro_drafter(macro_expert, tok),
    }
    router = _make_router(["copy", "macro"], {2: 40.0, 66: 1320.0})
    spec = DagSpeculator(qwen_target, router, experts, {"copy": 4, "macro": 2})
    pids = _pids(tok, REPETITIVE)
    r1 = spec.generate(pids, 24)
    r2 = spec.generate(pids, 24)
    assert r1["tokens"] == r2["tokens"]


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_dag_two_experts_accept_more_than_copy_alone(qwen_target) -> None:
    """Thesis probe: the union of copy+macro must accept >= copy alone."""
    from pathlib import Path

    from familydraft.experts.macro_render import build_renderer_from_config

    tok = qwen_target.tokenizer
    renderer = build_renderer_from_config(Path(__file__).parent.parent, tok, tok.vocab_size)
    macro_expert = MacroExpert(renderer, head=None)
    ce = CopyExpert(seed=4, min_length=3)

    pids = _pids(tok, REPETITIVE)

    spec_copy = DagSpeculator(
        qwen_target,
        _make_router(["copy"], {2: 40.0, 66: 1320.0}),
        {"copy": make_copy_drafter(ce, 4)},
        {"copy": 4},
    )
    spec_dag = DagSpeculator(
        qwen_target,
        _make_router(["copy", "macro"], {2: 40.0, 66: 1320.0}),
        {
            "copy": make_copy_drafter(ce, 4),
            "macro": make_macro_drafter(macro_expert, tok),
        },
        {"copy": 4, "macro": 2},
    )
    r_copy = spec_copy.generate(pids, 32)
    r_dag = spec_dag.generate(pids, 32)
    assert r_dag["tokens_per_round"] >= r_copy["tokens_per_round"]
