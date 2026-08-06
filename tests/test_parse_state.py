"""Goldens for parse_state v0 (plan todos 11/16)."""

from __future__ import annotations

from familydraft.experts.parse_state import parse_scan


def test_open_brace_yields_closer_candidates() -> None:
    state = parse_scan('def f():\n    return {"a": 1')
    assert state.bracket_stack == ("{",)
    assert "}" in state.candidates
    assert "\n}" in state.candidates


def test_nested_brackets_close_in_reverse() -> None:
    state = parse_scan("values = [{")
    assert state.bracket_stack == ("[", "{")
    assert "}]}"[:2] in state.candidates  # innermost closes first
    assert "\n}]" in state.candidates


def test_code_fence_membership() -> None:
    inside = parse_scan("```python\nprint(1)\n")
    assert inside.in_code_fence
    assert "```" in inside.candidates
    closed = parse_scan("```python\nprint(1)\n```")
    assert not closed.in_code_fence


def test_colon_line_yields_indented_continuation() -> None:
    text = "if ok:\n    x = 1\nif again:"
    state = parse_scan(text)
    assert "\n" + " " * 4 in state.candidates


def test_bullet_enumeration_continuation() -> None:
    state = parse_scan("items:\n- first")
    assert state.prev_line_is_bullet
    assert "\n- " in state.candidates


def test_numbered_enumeration_continuation() -> None:
    state = parse_scan("steps:\n1. draft")
    assert state.prev_line_numbered_end == 1
    assert "\n2. " in state.candidates


def test_plain_text_has_no_structure() -> None:
    state = parse_scan("the quick brown fox jumps over the lazy dog")
    assert state.bracket_stack == ()
    assert state.candidates == ()
    assert not state.in_code_fence


def test_bracket_inside_quotes_is_ignored() -> None:
    state = parse_scan('msg = "value [ here"')
    assert state.bracket_stack == ()
