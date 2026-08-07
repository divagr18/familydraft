"""Macro expert + parser features + macro head classifier (plan todo 16).

The macro expert proposes structural continuations: a small classifier head over
(trunk hidden state + parser features) scores the 64-action macro vocabulary,
and the top actions are rendered to family token ids via MacroRenderer. Parser
features consume parse_state.py (built in todo 11) - no rebuild. No learned or
discovered macros in Phase 1; the vocabulary is fixed.
"""

from __future__ import annotations

import torch
from torch import nn

from familydraft.experts.parse_state import parse_scan

PARSER_FEATURE_DIM = 6


def parser_features_from_text(text: str) -> list[float]:
    state = parse_scan(text)
    return [
        min(len(state.bracket_stack) / 8.0, 1.0),
        1.0 if state.in_code_fence else 0.0,
        min(state.current_indent / 16.0, 1.0),
        1.0 if state.prev_line_is_bullet else 0.0,
        1.0 if state.prev_line_numbered_end > 0 else 0.0,
        min(len(state.candidates) / 4.0, 1.0),
    ]


class MacroHead(nn.Module):
    def __init__(self, hidden_size: int, num_macros: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + PARSER_FEATURE_DIM, 128),
            nn.GELU(),
            nn.Linear(128, num_macros),
        )

    def forward(self, trunk_hidden: torch.Tensor, parser_feats: torch.Tensor) -> torch.Tensor:
        if trunk_hidden.dim() == 1:
            trunk_hidden = trunk_hidden.unsqueeze(0)
        if parser_feats.dim() == 1:
            parser_feats = parser_feats.unsqueeze(0)
        x = torch.cat([trunk_hidden, parser_feats.to(trunk_hidden.dtype)], dim=-1)
        return self.net(x)


class MacroExpert:
    def __init__(
        self, renderer, head: MacroHead | None = None, confidence_floor: float = 0.0
    ) -> None:
        self.renderer = renderer
        self.head = head
        self.confidence_floor = confidence_floor

    def propose_random_init(
        self, parser_feats: torch.Tensor, top_k: int = 1
    ) -> list[tuple[int, list[int]]]:
        """Head-free fallback: uniform scores over macros (used before training)."""
        scores = torch.zeros(self.renderer.num_macros)
        scores[:top_k] = 1.0
        top = torch.topk(scores, top_k).indices.tolist()
        return [(i, self.renderer.render_index(i)) for i in top]

    def propose(
        self,
        trunk_hidden: torch.Tensor | None,
        parser_feats: torch.Tensor,
        top_k: int = 1,
    ) -> list[tuple[int, list[int]]]:
        if self.head is None or trunk_hidden is None:
            return self.propose_random_init(parser_feats, top_k)
        with torch.inference_mode():
            logits = self.head(trunk_hidden, parser_feats)[0]
        top = torch.topk(logits, top_k).indices.tolist()
        return [(i, self.renderer.render_index(i)) for i in top]

    def propose_from_text(
        self,
        text: str,
        trunk_hidden: torch.Tensor | None = None,
        top_k: int = 1,
    ) -> list[tuple[int, list[int]]]:
        feats = torch.tensor(parser_features_from_text(text), dtype=torch.float32)
        return self.propose(trunk_hidden, feats, top_k)
