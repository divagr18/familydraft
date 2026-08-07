"""Trunk tests (plan todo 14). CPU-safe: loads Qwen3-0.6B in fp32 on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from familydraft.draft.trunk import Trunk, build_attribute_table

REPO = Path(__file__).parent.parent
MODEL = "Qwen/Qwen3-0.6B"
PARAM_CAP = 200_000_000


@pytest.fixture(scope="module")
def attributes() -> torch.Tensor:
    return build_attribute_table(REPO)


@pytest.fixture(scope="module")
def trunk(attributes) -> Trunk:
    return Trunk(MODEL, 6, attributes, dtype=torch.float32)


def test_forward_shape_matches_hidden(trunk) -> None:
    input_ids = torch.randint(0, 1000, (2, 16))
    out = trunk(input_ids, target_id=0)
    assert out.shape == (2, 16, trunk.hidden_size)
    assert trunk.hidden_size == 1024


def test_target_variant_embedding_distinct_and_trainable(trunk) -> None:
    z0 = trunk.target_variant(0)
    z2 = trunk.target_variant(2)
    assert z0.shape == (trunk.hidden_size,)
    assert not torch.allclose(z0, z2)
    trunk.target_variant.zero_grad()
    trunk.target_variant(1).sum().backward()
    grad = trunk.target_variant.residual.weight.grad
    assert grad is not None
    assert grad[1].abs().sum().item() > 0.0


def test_owned_params_under_cap(trunk) -> None:
    owned = trunk.owned_param_count()
    assert owned <= PARAM_CAP, f"owned params {owned} exceed cap {PARAM_CAP}"


def test_unknown_target_id_raises_keyerror(trunk) -> None:
    with pytest.raises(KeyError):
        trunk.target_variant(999)


def test_excessive_layers_rejected(attributes) -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        Trunk(MODEL, 100, attributes, dtype=torch.float32)
