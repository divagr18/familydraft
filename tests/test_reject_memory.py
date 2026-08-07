"""Rejection-memory expert tests (plan todo 18)."""

from __future__ import annotations

import pytest

from familydraft.experts.reject_memory import (
    MemoryLoadError,
    RejectionMemory,
)

FP_BASE = ((1, 2, 3, 4, 5, 6, 7, 8), "code", "Qwen/Qwen3-8B", "greedy")


def test_concept_scenario_repairs_after_three_rejections() -> None:
    mem = RejectionMemory(min_support=3)
    for _ in range(3):
        mem.record_rejection(FP_BASE, replacement=[11, 10], accepted_suffix=[])
    proposed = mem.propose(FP_BASE)
    assert proposed == [11, 10]
    rewritten = mem.rewrite([10], FP_BASE)
    assert rewritten == [11, 10]


def test_below_min_support_does_not_activate() -> None:
    mem = RejectionMemory(min_support=3)
    for _ in range(2):
        mem.record_rejection(FP_BASE, replacement=[11, 10])
    assert mem.propose(FP_BASE) is None
    assert mem.rewrite([10], FP_BASE) == [10]


def test_decay_expires_stale_entries() -> None:
    mem = RejectionMemory(min_support=3, half_life=2.0)
    for _ in range(4):
        mem.record_rejection(FP_BASE, replacement=[11, 10])
    assert mem.propose(FP_BASE) == [11, 10]
    mem.advance_events(100)
    assert mem.propose(FP_BASE) is None


def test_target_id_isolation() -> None:
    mem = RejectionMemory(min_support=1)
    fp_a = ((9, 9), "code", "target-A", "greedy")
    fp_b = ((9, 9), "code", "target-B", "greedy")
    mem.record_rejection(fp_a, replacement=[7, 7])
    assert mem.propose(fp_a) == [7, 7]
    assert mem.propose(fp_b) is None


def test_lru_eviction_is_deterministic_and_bounded() -> None:
    mem = RejectionMemory(min_support=1, max_entries=3)
    fps = [((i,), "code", "t", "greedy") for i in range(4)]
    for fp in fps:
        mem.record_rejection(fp, replacement=[fp[0][0]])
    assert mem.size == 3
    assert mem.propose(fps[0]) is None
    assert mem.propose(fps[3]) == [3]


def test_malformed_fingerprint_rejected() -> None:
    mem = RejectionMemory()
    with pytest.raises(ValueError, match="malformed fingerprint"):
        mem.record_rejection((1, 2, 3), replacement=[1])
    too_many = (tuple(range(9)), "code", "t", "greedy")
    with pytest.raises(ValueError, match="malformed fingerprint"):
        mem.record_rejection(too_many, replacement=[1])
    with pytest.raises(ValueError, match="malformed fingerprint"):
        mem.get_correction(("a", "b"))


def test_save_load_round_trip(tmp_path) -> None:
    mem = RejectionMemory(min_support=2)
    mem.record_rejection(FP_BASE, replacement=[5, 6])
    mem.record_rejection(FP_BASE, replacement=[5, 6])
    path = tmp_path / "memory.json"
    mem.save(path)
    restored = RejectionMemory.load(path)
    assert restored.propose(FP_BASE) == [5, 6]
    assert restored.size == 1


def test_corrupted_file_raises_with_name(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{{not json", encoding="utf-8")
    with pytest.raises(MemoryLoadError, match="bad.json"):
        RejectionMemory.load(path)


def test_load_or_empty_falls_back(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("corrupt", encoding="utf-8")
    mem = RejectionMemory.load_or_empty(path)
    assert mem.size == 0
    missing = RejectionMemory.load_or_empty(tmp_path / "does_not_exist.json")
    assert missing.size == 0
