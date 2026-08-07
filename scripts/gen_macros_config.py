"""Generate configs/macros.json - the 64-action macro vocabulary v1 (plan todo 16).

Superset of concept note section 4.4. Every macro has a deterministic render
text the MacroRenderer tokenizes. Parameterized actions (COPY_IDENTIFIER,
REPEAT_LINE_PREFIX) carry a canonical render here; context-dependent filling
is delegated to the copy expert at propose time.
"""

from __future__ import annotations

import json
from pathlib import Path

MACROS: list[tuple[str, str, str]] = [
    ("CLOSE_PAREN", ")", "closer"),
    ("CLOSE_BRACKET", "]", "closer"),
    ("CLOSE_BRACE", "}", "closer"),
    ("CLOSE_PAREN_NL", ")\n", "closer"),
    ("CLOSE_BRACKET_NL", "]\n", "closer"),
    ("CLOSE_BRACE_NL", "}\n", "closer"),
    ("CLOSE_PAREN_SEMI", ");", "closer"),
    ("CLOSE_BRACKET_SEMI", "];", "closer"),
    ("CLOSE_BRACE_SEMI", "};", "closer"),
    ("CLOSE_PAREN_COLON", "):", "closer"),
    ("CLOSE_PAREN_COMMA", "), ", "closer"),
    ("CLOSE_BRACKET_COMMA", "], ", "closer"),
    ("CLOSE_BRACE_COMMA", "}, ", "closer"),
    ("DOUBLE_CLOSE_PAREN", "))", "closer"),
    ("DOUBLE_CLOSE_BRACKET", "]]", "closer"),
    ("DOUBLE_CLOSE_BRACE", "}}", "closer"),
    ("OPEN_BRACE_NL", "{\n", "json"),
    ("OPEN_BRACKET_NL", "[\n", "json"),
    ("OPEN_BRACE", " {", "json"),
    ("OPEN_BRACKET", " [", "json"),
    ("OPEN_PAREN", " (", "json"),
    ("EMPTY_OBJECT", "{}", "json"),
    ("EMPTY_ARRAY", "[]", "json"),
    ("JSON_COLON", "\": ", "json"),
    ("JSON_STR_COMMA", "\",\n", "json"),
    ("JSON_OBJ_COMMA", "},\n", "json"),
    ("JSON_ARR_COMMA", "],\n", "json"),
    ("NEWLINE", "\n", "whitespace"),
    ("DOUBLE_NEWLINE", "\n\n", "whitespace"),
    ("NEWLINE_INDENT", "\n    ", "whitespace"),
    ("INDENT", "    ", "whitespace"),
    ("DOUBLE_INDENT", "        ", "whitespace"),
    ("SPACE", " ", "whitespace"),
    ("COLON", ":", "code"),
    ("COLON_NL_INDENT", ":\n    ", "code"),
    ("RETURN", "\nreturn ", "code"),
    ("IMPORT", "\nimport ", "code"),
    ("DEF", "\ndef ", "code"),
    ("CLASS", "\nclass ", "code"),
    ("COMMENT", "\n# ", "code"),
    ("ARROW", " -> ", "code"),
    ("LAMBDA", "lambda ", "code"),
    ("ASSIGN", " = ", "code"),
    ("COMMA", ",", "code"),
    ("SEMICOLON", ";", "code"),
    ("DOT", ".", "code"),
    ("ELLIPSIS", "...", "code"),
    ("CLOSE_CODE_FENCE", "\n```\n", "fence"),
    ("OPEN_CODE_FENCE_PY", "\n```python\n", "fence"),
    ("INLINE_CODE_FENCE", "```", "fence"),
    ("CONTINUE_BULLET", "\n- ", "enumeration"),
    ("CONTINUE_BULLET_STAR", "\n* ", "enumeration"),
    ("CONTINUE_NUMBERED_1", "\n1. ", "enumeration"),
    ("CONTINUE_NUMBERED_2", "\n2. ", "enumeration"),
    ("CLOSE_BLOCK", "\n", "enumeration"),
    ("COPY_IDENTIFIER", "value", "special"),
    ("REPEAT_LINE_PREFIX", "\n", "special"),
    ("TRUE", "true", "literal"),
    ("FALSE", "false", "literal"),
    ("NULL", "null", "literal"),
    ("NONE", "None", "literal"),
    ("OPEN_PAREN_SPACE", "( ", "closer"),
    ("COLON_SPACE", ": ", "code"),
    ("NEWLINE_CLOSE_BRACE", "\n}", "closer"),
]


def main() -> None:
    if len(MACROS) != 64:
        raise SystemExit(f"expected 64 macros, got {len(MACROS)}")
    names = [m[0] for m in MACROS]
    if len(set(names)) != 64:
        raise SystemExit("duplicate macro names")
    payload = {
        "schema": "familydraft.macros.v1",
        "count": 64,
        "macros": [
            {"name": n, "render": r, "category": c} for n, r, c in MACROS
        ],
    }
    out = Path("configs/macros.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(MACROS)} macros")


if __name__ == "__main__":
    main()
