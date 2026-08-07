"""Macro expert tests (plan todo 16)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from familydraft.experts.macro import MacroExpert, MacroHead, parser_features_from_text
from familydraft.experts.macro_render import (
    MacroDefinitionError,
    MacroRenderer,
    build_renderer_from_config,
)

REPO = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "goldens" / "macro_renders.json"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return build_renderer_from_config(REPO, tokenizer, tokenizer.vocab_size)


def test_exactly_64_macros(renderer) -> None:
    assert renderer.num_macros == 64


def test_all_renders_match_golden_and_in_vocab(renderer, tokenizer) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(golden.keys()) == set(renderer.names)
    for name in renderer.names:
        ids = renderer.render(name)
        assert ids == golden[name], f"render drift for {name}"
        for tok in ids:
            assert 0 <= tok < tokenizer.vocab_size, f"{name} token {tok} out of vocab"


def test_render_retokenize_round_trip(renderer, tokenizer) -> None:
    for name in renderer.names:
        ids = renderer.render(name)
        if not ids:
            continue
        text = tokenizer.decode(ids, skip_special_tokens=False)
        re_ids = tokenizer.encode(text, add_special_tokens=False)
        assert re_ids == ids, f"token-boundary drift for {name}: {ids} vs {re_ids}"


def test_macro_head_forward_shape() -> None:
    head = MacroHead(hidden_size=1024, num_macros=64)
    trunk = torch.randn(2, 1024)
    feats = torch.rand(2, 6)
    logits = head(trunk, feats)
    assert logits.shape == (2, 64)


def test_macro_expert_propose_returns_valid_renders(renderer) -> None:
    expert = MacroExpert(renderer, head=None)
    proposals = expert.propose_from_text('{"a": 1,', top_k=3)
    assert len(proposals) == 3
    for idx, ids in proposals:
        assert 0 <= idx < 64
        assert ids == renderer.render_index(idx)


def test_parser_features_vector_shape() -> None:
    feats = parser_features_from_text('{"a": [1, 2')
    assert len(feats) == 6
    assert all(isinstance(f, float) for f in feats)


def test_label_deriver_reproduces_golden_macros(renderer) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "derive_macro_labels", REPO / "scripts" / "derive_macro_labels.py"
    )
    dml = importlib.util.module_from_spec(spec)
    sys.modules["derive_macro_labels"] = dml
    spec.loader.exec_module(dml)
    table, max_len = dml.build_render_table(renderer)

    cases = [
        "NEWLINE_INDENT",
        "CONTINUE_BULLET",
        "CLOSE_CODE_FENCE",
        "RETURN",
        "JSON_COLON",
        "EMPTY_ARRAY",
        "ASSIGN",
        "DOUBLE_NEWLINE",
    ]
    for name in cases:
        continuation = renderer.render(name)
        labels = dml.derive_labels(continuation, table, max_len)
        assert labels[0] == renderer.index_of(name), f"{name} mislabeled as {labels[0]}"


def test_malformed_macro_entry_rejected_with_name(tokenizer) -> None:
    with pytest.raises(MacroDefinitionError, match="MISSING_RENDER"):
        MacroRenderer(tokenizer, [{"name": "MISSING_RENDER"}], tokenizer.vocab_size)
    with pytest.raises(MacroDefinitionError, match="BAD_TYPE"):
        MacroRenderer(
            tokenizer, [{"name": "BAD_TYPE", "render": 123}], tokenizer.vocab_size
        )


def test_drafting_call_latency_within_budget(renderer) -> None:
    cost_curve_path = REPO / "runs" / "microbench" / "cost_curve.json"
    if not cost_curve_path.exists():
        pytest.skip("cost_curve.json not generated")
    record = json.loads(cost_curve_path.read_text(encoding="utf-8"))
    budget_ms = 0.1 * record["decode_ms_per_token"]
    expert = MacroExpert(renderer, head=None)
    text = '{"key": [1, 2,'
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        expert.propose_from_text(text, top_k=2)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    p50_ms = latencies[len(latencies) // 2] * 1000
    assert p50_ms <= budget_ms, f"p50 {p50_ms:.3f}ms exceeds budget {budget_ms:.3f}ms"
