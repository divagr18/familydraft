"""General neural continuation expert (plan todo 15).

An autoregressive LM head over the shared trunk state. The trunk (a truncated
family member) supplies hidden states; this expert adds the output projection
over the full family vocabulary. The shared input embedding stays frozen; the
LM head is a separately initialised trainable projection so distillation does
not mutate the family embedding.
"""

from __future__ import annotations

import torch
from torch import nn


class GeneralExpert(nn.Module):
    def __init__(self, trunk) -> None:
        super().__init__()
        self.trunk = trunk
        hidden = trunk.hidden_size
        vocab = trunk.backbone.embed_tokens.num_embeddings
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(trunk.backbone.embed_tokens.weight)
        self.lm_head = self.lm_head.to(trunk.backbone.embed_tokens.weight.dtype)
        self.freeze_shared_embedding()

    def freeze_shared_embedding(self) -> None:
        self.trunk.backbone.embed_tokens.weight.requires_grad_(False)

    @property
    def vocab_size(self) -> int:
        return self.lm_head.out_features

    def forward(self, input_ids: torch.Tensor, target_id: int) -> torch.Tensor:
        h = self.trunk(input_ids, target_id)
        return self.lm_head(h)

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_id: int,
        prompt_len: int,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        logits = self.forward(input_ids, target_id)
        full_len = input_ids.shape[1]
        shift_logits = logits[:, prompt_len - 1 : full_len - 1, :].float()
        shift_labels = input_ids[:, prompt_len:full_len]
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return loss_fn(
            shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1)
        )


def make_random_trunk_like(trunk):
    """Control trunk: same shape as `trunk` but with re-initialised weights."""
    import copy

    control = copy.deepcopy(trunk)
    for module in control.backbone.layers:
        for p in module.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    torch.nn.init.normal_(control.backbone.embed_tokens.weight, std=0.02)
    return control
