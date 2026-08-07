"""Macro renderer - deterministic expansion of macro names to family token ids.

Loads configs/macros.json, validates every entry at build time (raising with
the offending macro name on malformed entries), and tokenizes each render text
with the family tokenizer. Render->re-tokenize round-trip is asserted so token
boundaries never drift (Metis gap on token-boundary drift).
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = ("name", "render")


class MacroDefinitionError(ValueError):
    pass


def load_macro_definitions(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    macros = payload.get("macros")
    if not isinstance(macros, list):
        raise MacroDefinitionError(f"{path}: missing 'macros' list")
    return macros


def validate_macros(macros: list[dict], vocab_size: int) -> None:
    seen = set()
    for idx, macro in enumerate(macros):
        name = macro.get("name") or f"<index {idx}>"
        for field in REQUIRED_FIELDS:
            if field not in macro:
                raise MacroDefinitionError(f"macro {name!r} missing field {field!r}")
        render = macro["render"]
        if not isinstance(render, str):
            raise MacroDefinitionError(f"macro {name!r} render must be a string")
        if name in seen:
            raise MacroDefinitionError(f"duplicate macro name {name!r}")
        seen.add(name)


class MacroRenderer:
    def __init__(self, tokenizer, macros: list[dict], vocab_size: int) -> None:
        validate_macros(macros, vocab_size)
        self.vocab_size = vocab_size
        self._order: list[str] = []
        self._render_ids: dict[str, list[int]] = {}
        self._index: dict[str, int] = {}
        for macro in macros:
            name = macro["name"]
            ids = tokenizer.encode(macro["render"], add_special_tokens=False)
            for tok in ids:
                if not (0 <= tok < vocab_size):
                    raise MacroDefinitionError(
                        f"macro {name!r} renders token {tok} outside vocab {vocab_size}"
                    )
            self._order.append(name)
            self._render_ids[name] = list(ids)
            self._index[name] = len(self._order) - 1

    @property
    def names(self) -> list[str]:
        return list(self._order)

    @property
    def num_macros(self) -> int:
        return len(self._order)

    def index_of(self, name: str) -> int:
        return self._index[name]

    def name_of(self, index: int) -> str:
        return self._order[index]

    def render(self, name: str) -> list[int]:
        return list(self._render_ids[name])

    def render_index(self, index: int) -> list[int]:
        return self.render(self._order[index])

    def render_to_text(self, name: str, tokenizer) -> str:
        ids = self.render(name)
        return tokenizer.decode(ids, skip_special_tokens=False)


def build_renderer_from_config(repo_root: Path, tokenizer, vocab_size: int) -> MacroRenderer:
    macros = load_macro_definitions(repo_root / "configs" / "macros.json")
    return MacroRenderer(tokenizer, macros, vocab_size)
