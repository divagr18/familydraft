"""Shared trunk + target-variant conditioning (plan todo 14).

A truncated member of the target family produces the shared hidden state h_t
consumed by the neural experts and the router. Target identity conditions the
trunk through an ATTRIBUTE-CONDITIONED embedding: an MLP over a normalized
target-config attribute vector (so UNSEEN targets can interpolate) plus a
learned per-id residual for seen targets. The input embedding is shared with
the family (already resident for serving); the trunk's owned parameters are the
truncated layers plus the target-variant module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

ATTR_DIM = 5


def normalize_attributes(raw: dict) -> list[float]:
    return [
        math.log1p(raw["params_b"]) / math.log1p(40.0),
        raw["layers"] / 64.0,
        raw["hidden"] / 8192.0,
        math.log1p(raw["active_b"]) / math.log1p(40.0),
        float(raw["moe"]),
    ]


def build_attribute_table(repo_root: Path) -> torch.Tensor:
    attrs = json.loads((repo_root / "configs" / "target_attributes.json").read_text("utf-8"))
    ids = json.loads((repo_root / "configs" / "target_ids.json").read_text("utf-8"))
    num_ids = max(info["id"] for info in ids.values()) + 1
    table = torch.zeros(num_ids, ATTR_DIM)
    for repo, info in ids.items():
        if repo in attrs:
            table[info["id"]] = torch.tensor(normalize_attributes(attrs[repo]))
    return table


class TargetVariantEmbedding(nn.Module):
    def __init__(self, hidden: int, attributes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("attributes", attributes)
        self.mlp = nn.Sequential(
            nn.Linear(attributes.shape[1], hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.residual = nn.Embedding(attributes.shape[0], hidden)
        nn.init.zeros_(self.residual.weight)
        self.enabled = True

    def forward(self, target_id: int) -> torch.Tensor:
        if not self.enabled:
            dtype = self.mlp[0].weight.dtype
            return torch.zeros(
                self.residual.embedding_dim, device=self.attributes.device, dtype=dtype
            )
        if target_id < 0 or target_id >= self.attributes.shape[0]:
            raise KeyError(self._valid_message(target_id))
        attr = self.attributes[target_id]
        idx = torch.tensor(target_id, device=self.attributes.device)
        return self.mlp(attr) + self.residual(idx)

    def _valid_message(self, target_id: int) -> str:
        return f"unknown target_id {target_id}; valid range 0..{self.attributes.shape[0]-1}"


class Trunk(nn.Module):
    def __init__(
        self,
        model_repo: str,
        layers: int,
        attributes: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM

        full = AutoModelForCausalLM.from_pretrained(model_repo, torch_dtype=dtype)
        backbone = full.model
        available = len(backbone.layers)
        if layers > available:
            raise ValueError(
                f"layers={layers} exceeds available {available} for {model_repo} "
                "(would exceed the trunk budget)"
            )
        backbone.layers = backbone.layers[:layers]
        self.backbone = backbone
        self.hidden_size = backbone.embed_tokens.embedding_dim
        self.num_layers = layers
        self.target_variant = TargetVariantEmbedding(self.hidden_size, attributes)
        del full.lm_head

    def owned_param_count(self) -> int:
        total = sum(p.numel() for p in self.backbone.layers.parameters())
        total += sum(p.numel() for p in self.target_variant.parameters())
        return total

    def forward(self, input_ids: torch.Tensor, target_id: int) -> torch.Tensor:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        z = self.target_variant(target_id).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + z
        out = self.backbone(inputs_embeds=inputs_embeds)
        return out.last_hidden_state


def build_trunk_from_config(repo_root: Path) -> Trunk:
    import yaml

    cfg = yaml.safe_load((repo_root / "configs" / "trunk.yaml").read_text("utf-8"))
    attributes = build_attribute_table(repo_root)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]
    return Trunk(cfg["model_repo"], cfg["layers"], attributes, dtype)
