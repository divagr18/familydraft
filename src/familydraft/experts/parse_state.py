"""Lightweight structural parse state (v0, plan todos 11/16).

Pure-Python deterministic state machine over generated text. Tracks bracket
stack, quote state, code-fence membership, line indentation and enumeration
structure, and derives candidate macro continuations (the text spans a
structural expert would propose). No external parser dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}
_FENCE = "```"


@dataclass(frozen=True)
class ParseState:
    bracket_stack: tuple[str, ...]
    in_single_quote: bool
    in_double_quote: bool
    in_code_fence: bool
    fence_line_start: bool
    current_indent: int
    last_line_stripped: str
    prev_line_is_bullet: bool
    prev_line_numbered_end: int
    text_len: int
    candidates: tuple[str, ...] = field(default=())


def _scan(text: str) -> tuple[list[str], bool, bool, bool]:
    """Track bracket/quote state. Limitation (v0): delimiters inside code
    fences are not fenced off; documented approximation for oracle use."""
    stack: list[str] = []
    in_sq = False
    in_dq = False
    for ch in text:
        if in_sq:
            if ch == "'":
                in_sq = False
            continue
        if in_dq:
            if ch == '"':
                in_dq = False
            continue
        if ch == "'":
            in_sq = True
        elif ch == '"':
            in_dq = True
        elif ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if stack and stack[-1] == _CLOSE_TO_OPEN[ch]:
                stack.pop()
    fence_count = text.count(_FENCE)
    in_fence = fence_count % 2 == 1
    return stack, in_sq, in_dq, in_fence


def _derive_candidates(
    bracket_stack: list[str],
    in_fence: bool,
    last_line: str,
    indent: int,
    prev_bullet: bool,
    prev_numbered: int,
) -> list[str]:
    out: list[str] = []
    if bracket_stack:
        closers = "".join(_OPEN_TO_CLOSE[b] for b in reversed(bracket_stack))
        out.append(closers)
        out.append("\n" + closers)
    if in_fence:
        out.append("```")
        out.append("\n```")
    stripped = last_line.rstrip()
    if stripped.endswith(":"):
        out.append("\n" + " " * (indent + 4))
    if prev_bullet:
        out.append("\n- ")
    if prev_numbered > 0:
        out.append(f"\n{prev_numbered + 1}. ")
    return out


def _json_candidates(text: str) -> list[str]:
    """Predict likely next structural tokens of a JSON document from prefix text.

    JSON tokenizes structural boundaries into repeating multi-char tokens
    ('":', ' "', '",\\n'), one per key/value boundary. A tiny state machine
    over the prefix predicts which boundary comes next so the macro mechanism
    can credit structured continuations. Heuristic; returns candidate strings
    that may match a token's decoded text.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    str_role = "value"
    state = "start"
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                state = "after_key" if str_role == "key" else "after_value"
            continue
        if ch == '"':
            in_str = True
            if stack and stack[-1] == "{" and state in ("after_open", "after_comma"):
                str_role = "key"
                state = "in_key"
            else:
                str_role = "value"
                state = "in_value"
            continue
        if ch in "{[":
            stack.append(ch)
            state = "after_open"
        elif ch in "}]":
            if stack:
                stack.pop()
            state = "after_value"
        elif ch == ":":
            state = "after_colon"
        elif ch == ",":
            state = "after_comma"
        elif not ch.isspace():
            state = "in_value"
    if in_str:
        # Right after an opening quote (prefix ends with '"') the string body
        # comes next and is unpredictable. Mid-string, a key's next structural
        # token is the closing-quote-plus-colon merge.
        if text.endswith('"') and not text.endswith('\\"'):
            return []
        if str_role == "key":
            return ['":', '": ']
        return []
    top = stack[-1] if stack else None
    if state == "after_colon":
        return [' "', " {", " [", " "]
    if state == "after_open":
        return [' "', "}", "{", "["] if top == "{" else ["{", "[", '"', "]"]
    if state == "after_comma":
        return [' "', "\n  ", ", "] if top == "[" else [' "', "\n  "]
    if state in ("after_value", "after_key"):
        return [",", "}", "]", '",\n', ",\n"]
    return []


def parse_scan(text: str) -> ParseState:
    stack, in_sq, in_dq, in_fence = _scan(text)
    lines = text.split("\n")
    last_line = lines[-1] if lines else ""
    indent = len(last_line) - len(last_line.lstrip(" "))
    active_line = ""
    for line in reversed(lines):
        if line.strip():
            active_line = line.strip()
            break
    prev_bullet = active_line.startswith("- ") or active_line.startswith("* ")
    prev_numbered = 0
    if active_line and active_line[0].isdigit():
        head = ""
        for ch in active_line:
            if ch.isdigit():
                head += ch
            else:
                break
        if active_line[len(head) : len(head) + 2].startswith(". "):
            prev_numbered = int(head)
    fence_line_start = in_fence and not text.rstrip("\n").endswith(" ")
    candidates = _derive_candidates(stack, in_fence, last_line, indent, prev_bullet, prev_numbered)
    candidates.extend(_json_candidates(text))
    return ParseState(
        bracket_stack=tuple(stack),
        in_single_quote=in_sq,
        in_double_quote=in_dq,
        in_code_fence=in_fence,
        fence_line_start=fence_line_start,
        current_indent=indent,
        last_line_stripped=last_line.rstrip(),
        prev_line_is_bullet=prev_bullet,
        prev_line_numbered_end=prev_numbered,
        text_len=len(text),
        candidates=tuple(candidates),
    )
