"""Integrated draft->verify loop tests (plan Wave E).

Two tiers of evidence:

1. Exact-numerics tier (tiny fp32 model): end-to-end byte-for-byte losslessness
   for empty / random / oracle / copy drafters. fp32 removes the bf16
   batch-vs-sequential numerics, so this isolates and proves the LOOP LOGIC
   (KV-cache cropping, acceptance indexing, bonus handling) is correct.

2. Real-model tier (Qwen3-0.6B bf16): determinism (same drafter+prompt ->
   identical output across reruns) and high agreement with vanilla. Byte-equality
   with standalone greedy is NOT asserted here because batched bf16 verification
   accumulates KV numerics drift that can flip near-tie argmaxes (a numerical
   artifact inherent to batched speculation, not a logic bug in this loop).
"""

from __future__ import annotations

import random
import types

import pytest
import torch

from familydraft.eval.draft_loop import (
    IntegratedSpeculator,
    make_copy_drafter,
)
from familydraft.experts.copy import CopyExpert

CUDA = torch.cuda.is_available()

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Write a Python function that returns the sum of a list of numbers:"
REPETITIVE_PROMPT = "Write 8 list append statements adding i to a list for i from 0 to 7:"


def _wrap(model, tokenizer):
    return types.SimpleNamespace(model=model, tokenizer=tokenizer)


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][
        0
    ].tolist()


def _greedy(model, tokenizer, prompt: str, n: int) -> list[int]:
    pids = _prompt_ids(tokenizer, prompt)
    with torch.inference_mode():
        out = model.generate(
            torch.tensor([pids], device=next(model.parameters()).device),
            max_new_tokens=n,
            do_sample=False,
        )
    return out[0, len(pids):].tolist()


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


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_exact_numerics_empty_drafter_lossless(tiny_target) -> None:
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, PROMPT, 20)
    pids = _prompt_ids(tiny_target.tokenizer, PROMPT)
    spec = IntegratedSpeculator(tiny_target, draft_fn=lambda ctx: [], spec_len=4)
    assert spec.generate(pids, 20)["tokens"] == vanilla


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_exact_numerics_random_drafter_lossless(tiny_target) -> None:
    rng = random.Random(0)

    def rand_draft(ctx):
        return [rng.randrange(0, 500) for _ in range(4)]

    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, PROMPT, 20)
    pids = _prompt_ids(tiny_target.tokenizer, PROMPT)
    spec = IntegratedSpeculator(tiny_target, rand_draft, spec_len=4)
    assert spec.generate(pids, 20)["tokens"] == vanilla


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_exact_numerics_oracle_drafter_fully_accepted(tiny_target) -> None:
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, PROMPT, 24)
    pids = _prompt_ids(tiny_target.tokenizer, PROMPT)

    def oracle_draft(ctx):
        i = len(ctx) - len(pids)
        return vanilla[i : i + 4]

    spec = IntegratedSpeculator(tiny_target, oracle_draft, spec_len=4)
    res = spec.generate(pids, 24)
    assert res["tokens"] == vanilla
    assert res["tokens_per_round"] > 1.5


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_exact_numerics_copy_drafter_lossless(tiny_target) -> None:
    copy_expert = CopyExpert(seed=4, min_length=3)
    vanilla = _greedy(tiny_target.model, tiny_target.tokenizer, REPETITIVE_PROMPT, 24)
    pids = _prompt_ids(tiny_target.tokenizer, REPETITIVE_PROMPT)
    spec = IntegratedSpeculator(tiny_target, make_copy_drafter(copy_expert, 4), spec_len=4)
    assert spec.generate(pids, 24)["tokens"] == vanilla


@pytest.fixture(scope="module")
def qwen_target():
    from familydraft.targets.wrapper import TargetModel

    return TargetModel.load(MODEL, dtype="bf16")


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_real_model_speculative_is_deterministic(qwen_target) -> None:
    copy_expert = CopyExpert(seed=4, min_length=3)
    pids = _prompt_ids(qwen_target.tokenizer, REPETITIVE_PROMPT)
    spec = IntegratedSpeculator(qwen_target, make_copy_drafter(copy_expert, 4), 4, 0)
    r1 = spec.generate(pids, 24)
    r2 = spec.generate(pids, 24)
    assert r1["tokens"] == r2["tokens"]


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_real_model_empty_drafter_matches_vanilla(qwen_target) -> None:
    pids = _prompt_ids(qwen_target.tokenizer, PROMPT)
    vanilla = qwen_target.generate_greedy(torch.tensor([pids]), 16)[0, len(pids):].tolist()
    spec = IntegratedSpeculator(qwen_target, draft_fn=lambda ctx: [], spec_len=4)
    assert spec.generate(pids, 16)["tokens"] == vanilla


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_real_model_agreement_is_high(qwen_target) -> None:
    copy_expert = CopyExpert(seed=4, min_length=3)
    pids = _prompt_ids(qwen_target.tokenizer, REPETITIVE_PROMPT)
    vanilla = qwen_target.generate_greedy(torch.tensor([pids]), 24)[0, len(pids):].tolist()
    spec = IntegratedSpeculator(qwen_target, make_copy_drafter(copy_expert, 4), 4, 0)
    res = spec.generate(pids, 24)
    agree = sum(1 for a, b in zip(res["tokens"], vanilla) if a == b)
    ratio = agree / len(vanilla)
    assert ratio >= 0.6, f"agreement too low: {ratio:.2f}"
