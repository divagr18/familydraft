"""Target-variant transfer harness tests (plan todo 24, locally-completable).

Acceptance criteria:
  - QA failure scenario: attempting to evaluate a target in the training set
    {4B, 8B, 14B} makes the harness REFUSE with the named target (guard test).
  - QA happy scenario: unseen eval targets pass the guard; CSV columns match
    the plan (target x task class: accepted length, tokens/sec, drafter
    overhead) plus config hash + seed for reproducibility.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

SCRIPT = _REPO / "scripts" / "run_transfer_eval.py"
TRAIN_TARGETS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
UNSEEN_TARGETS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-32B", "Qwen/Qwen3-Coder-30B-A3B"]


def _load_module():
    spec = importlib.util.spec_from_file_location("run_transfer_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guard_refuses_seen_target() -> None:
    mod = _load_module()
    try:
        mod._validate_eval_targets(["Qwen/Qwen3-8B"])
    except ValueError as exc:
        assert "REFUSED" in str(exc)
        assert "Qwen/Qwen3-8B" in str(exc)
    else:
        raise AssertionError("seen training target was not refused")


def test_guard_refuses_mixed_list_naming_offenders() -> None:
    mod = _load_module()
    try:
        mod._validate_eval_targets(["Qwen/Qwen3-0.6B", "Qwen/Qwen3-14B"])
    except ValueError as exc:
        assert "Qwen/Qwen3-14B" in str(exc)
    else:
        raise AssertionError("mixed list with a seen target was not refused")


def test_guard_passes_unseen_targets() -> None:
    mod = _load_module()
    cleaned = mod._validate_eval_targets(UNSEEN_TARGETS)
    assert cleaned == UNSEEN_TARGETS


def test_script_refuses_seen_target_via_cli() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_transfer_eval.py",
         "--eval-targets", "Qwen/Qwen3-8B"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    # Guard fires BEFORE the CUDA check, so this must exit 1 (not 2) and name
    # the offending target regardless of GPU availability.
    assert proc.returncode == 1, proc.stderr
    assert "REFUSED" in proc.stderr
    assert "Qwen/Qwen3-8B" in proc.stderr


def test_csv_columns_match_plan() -> None:
    mod = _load_module()
    assert mod.CSV_COLUMNS == [
        "target", "task_class", "accepted_length", "tokens_per_sec",
        "drafter_overhead_rounds", "config_hash", "seed",
    ]
    # plan: targets x task classes {code, structured}
    assert mod.TASK_CLASSES == ["code", "structured"]
    # training set per plan (zero-shot transfer: train on {4B,8B,14B} only)
    assert mod.TRAIN_TARGETS == TRAIN_TARGETS


def test_delta_computation_logs_unseen_minus_seen() -> None:
    mod = _load_module()
    rows = [
        {"target": "Qwen/Qwen3-32B", "task_class": "code", "accepted_length": 7.0},
        {"target": "Qwen/Qwen3-32B", "task_class": "structured", "accepted_length": 5.0},
    ]
    baseline = _REPO / "runs" / "results" / "phase2_train_eval.csv"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        "task_class,accepted_length\ncode,4.0\nstructured,3.0\n", encoding="utf-8"
    )
    try:
        deltas = mod._compute_delta(rows, baseline)
    finally:
        baseline.unlink(missing_ok=True)
    assert len(deltas) == 2
    assert deltas[0]["transfer_delta"] == 3.0  # 7.0 - 4.0
    assert deltas[1]["transfer_delta"] == 2.0  # 5.0 - 3.0
