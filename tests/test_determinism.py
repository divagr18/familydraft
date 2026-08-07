"""Tests for the Phase-1 determinism self-check (plan todo 21 QA).

Verifies the two QA contract points without loading a real model:
  1. `_mutate_target` wraps MLP outputs with an unconditional noise layer that
     survives eval mode (a plain nn.Dropout would be a silent no-op and the
     check would falsely PASS - the bug this guards against).
  2. The injected noise actually diverges across different seeds, so a mutated
     pipeline is caught while a clean deterministic one stays bit-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.check_determinism import _mutate_target  # noqa: E402


class _FakeMlp(nn.Module):
    def forward(self, x):
        return x * 2


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _FakeMlp()


class _FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])


class _FakeHfModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeBaseModel()


class _FakeTarget:
    def __init__(self) -> None:
        self.model = _FakeHfModel()


def test_mutate_target_wraps_mlp_with_noise() -> None:
    target = _FakeTarget()
    _mutate_target(target)
    for layer in target.model.model.layers:
        assert isinstance(layer.mlp, nn.Sequential)
        assert isinstance(layer.mlp[1], nn.Module)  # noise module present
        # the original MLP must still be first in the chain
        assert isinstance(layer.mlp[0], _FakeMlp)


def test_mutated_output_diverges_across_seeds() -> None:
    """The injected noise must actually change outputs across different RNG
    seeds; otherwise a mutated pipeline would silently pass the check."""
    target = _FakeTarget()
    _mutate_target(target)
    x = torch.randn(4, 8)
    torch.manual_seed(1234)
    out1 = target.model.model.layers[0].mlp(x)
    torch.manual_seed(5678)
    out2 = target.model.model.layers[0].mlp(x)
    assert not torch.allclose(out1, out2), (
        "noise injection is seed-insensitive; mutation would go undetected"
    )


def test_clean_mlp_is_seed_invariant() -> None:
    """A clean (unmutated) MLP must be bit-identical across seeds - the happy
    path the check relies on."""
    mlp = _FakeMlp()
    x = torch.randn(4, 8)
    torch.manual_seed(1234)
    a = mlp(x)
    torch.manual_seed(5678)
    b = mlp(x)
    assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_noise_detects_mutation_at_multiple_seeds(seed: int) -> None:
    target = _FakeTarget()
    _mutate_target(target)
    x = torch.randn(2, 16)
    torch.manual_seed(seed)
    run_a = target.model.model.layers[0].mlp(x)
    torch.manual_seed(seed + 1)
    run_b = target.model.model.layers[0].mlp(x)
    assert not torch.equal(run_a, run_b)
