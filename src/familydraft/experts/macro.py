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
        self,
        parser_feats: torch.Tensor,
        top_k: int = 1,
        top_open: str | None = None,
    ) -> list[tuple[int, list[int]]]:
        """Head-free fallback: score macros from parser features instead of
        uniform scores. The bracket stack (feat 0), code fence (feat 1), indent
        (feat 2), bullet/numbered (feats 3-4) and JSON-candidate density (feat 5)
        select the structurally correct continuation."""
        scores = torch.zeros(self.renderer.num_macros)
        by_name = {name: self.renderer.index_of(name) for name in self.renderer.names}

        def add(name: str, weight: float) -> None:
            idx = by_name.get(name)
            if idx is not None:
                scores[idx] += weight

        feats = parser_feats.tolist()
        stack_depth = int(round(feats[0] * 8.0))
        in_fence = feats[1] >= 0.5
        indent = int(round(feats[2] * 16.0))
        is_bullet = feats[3] >= 0.5
        numbered = int(round(feats[4] * 4.0))
        json_density = feats[5]

        if in_fence:
            add("CLOSE_CODE_FENCE", 3.0)
        if stack_depth > 0:
            closer_by_open = {"(": "CLOSE_PAREN", "[": "CLOSE_BRACKET", "{": "CLOSE_BRACE"}
            closer = closer_by_open.get(top_open or "", "CLOSE_PAREN")
            add(closer, 3.0)
            if json_density > 0.75 and top_open in ("{", "["):
                add("CLOSE_BRACE", 2.0)
                add("CLOSE_BRACKET", 1.5)
            else:
                add("CLOSE_PAREN", 1.5)
                add("CLOSE_BRACKET", 1.5)
                add("CLOSE_BRACE", 1.5)
            if indent > 0:
                add("CLOSE_PAREN_NL", 1.0)
            add("CLOSE_PAREN_SEMI", 0.5)
            add("DOUBLE_CLOSE_PAREN", 0.5)
        if numbered > 0:
            add("CONTINUE_NUMBERED_1", 2.0)
        if is_bullet:
            add("CONTINUE_BULLET", 2.0)
        if indent > 0:
            add("NEWLINE_INDENT", 1.5)
        add("COLON", 0.5)
        add("COMMA", 0.5)
        if scores.sum() == 0:
            add("NEWLINE", 1.0)
        top = torch.topk(scores, min(top_k, self.renderer.num_macros)).indices.tolist()
        return [(i, self.renderer.render_index(i)) for i in top]

    def propose(
        self,
        trunk_hidden: torch.Tensor | None,
        parser_feats: torch.Tensor,
        top_k: int = 1,
        top_open: str | None = None,
    ) -> list[tuple[int, list[int]]]:
        if self.head is None or trunk_hidden is None:
            return self.propose_random_init(parser_feats, top_k, top_open)
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
        state = parse_scan(text)
        feats = torch.tensor(parser_features_from_text(text), dtype=torch.float32)
        top_open = state.bracket_stack[-1] if state.bracket_stack else None
        return self.propose(trunk_hidden, feats, top_k, top_open)
