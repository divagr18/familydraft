"""General expert tests (plan todo 15). CPU-safe on Qwen3-0.6B fp32."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from familydraft.experts.general import GeneralExpert, make_random_trunk_like

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def trunk():
    from familydraft.draft.trunk import build_trunk_from_config

    t = build_trunk_from_config(REPO)
    return t.to(torch.float32)


@pytest.fixture(scope="module")
def expert(trunk):
    return GeneralExpert(trunk)


def test_forward_shape_and_vocab(expert) -> None:
    input_ids = torch.randint(0, 1000, (1, 8))
    logits = expert(input_ids, target_id=0)
    assert logits.shape == (1, 8, expert.vocab_size)
    assert expert.vocab_size == 151936


def test_shared_embedding_frozen_head_trainable(expert) -> None:
    assert expert.trunk.backbone.embed_tokens.weight.requires_grad is False
    assert expert.lm_head.weight.requires_grad is True


def test_next_token_loss_is_finite_positive(expert) -> None:
    input_ids = torch.randint(0, 1000, (1, 12))
    loss = expert.next_token_loss(input_ids, target_id=0, prompt_len=4)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_conditioning_changes_logits(trunk) -> None:
    expert = GeneralExpert(trunk)
    input_ids = torch.randint(0, 1000, (1, 6))
    logits0 = expert(input_ids, target_id=0)
    logits2 = expert(input_ids, target_id=2)
    assert not torch.allclose(logits0, logits2)


def test_control_trunk_weights_differ(trunk) -> None:
    control = make_random_trunk_like(trunk)
    w_orig = trunk.backbone.layers[0].self_attn.q_proj.weight
    w_ctrl = control.backbone.layers[0].self_attn.q_proj.weight
    assert not torch.allclose(w_orig, w_ctrl)


def test_checkpoint_reload_produces_valid_ids(expert, tmp_path) -> None:
    ckpt = tmp_path / "expert.pt"
    torch.save(expert.state_dict(), ckpt)
    fresh = GeneralExpert(expert.trunk)
    fresh.load_state_dict(torch.load(ckpt, map_location="cpu"))
    input_ids = torch.randint(0, 1000, (1, 6))
    logits = fresh(input_ids, target_id=0)
    ids = torch.argmax(logits, dim=-1)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < fresh.vocab_size
