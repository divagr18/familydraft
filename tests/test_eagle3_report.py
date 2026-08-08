"""EAGLE-3 report schema-conformance tests (plan todo 13, locally-completable).

Acceptance criteria (plan todo 13):
  - report validates against configs/baseline_report.schema.json
  - runs >= 5 (schema minimum; the report producer defaults to 5)
  - flops_per_emitted_token > 0 (schema exclusiveMinimum 0) - computed
    structurally via src/familydraft/eval/flops.py, mirroring run_baselines
  - target-load failure fails honestly (no invalid placeholder row)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parent.parent
SCHEMA = _REPO / "configs" / "baseline_report.schema.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "eval_eagle3", _REPO / "scripts" / "eval_eagle3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**overrides) -> argparse.Namespace:
    base = {
        "repo": "Qwen/Qwen3-8B",
        "checkpoint": Path("runs/baselines/eagle3_Qwen-Qwen3-8B/checkpoints"),
        "specforge_sha": "7d5a693",
        "train_hours": 12.0,
        "task_class": "structured",
        "max_prompts": 8,
        "spec_len": 4,
        "tpr": 1.0,
        "runs": 5,
        "out": "",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_report_validates_against_schema() -> None:
    mod = _load_module()
    measured = {"acc_len": -1.0, "note": "pod-measured", "vanilla_tps_probe": 21.5}
    report = mod.build_report(measured, _args())
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)  # raises on violation
    assert mod.validate_report(report, SCHEMA) == []


def test_runs_minimum_five_enforced() -> None:
    """Schema requires runs >= 5; producer must never emit fewer."""
    mod = _load_module()
    report = mod.build_report({"vanilla_tps_probe": 10.0}, _args(runs=1))
    assert report["runs"] >= 5


def test_flops_per_emitted_token_positive() -> None:
    """Schema requires exclusiveMinimum 0 - the old 0.0 placeholder failed."""
    mod = _load_module()
    report = mod.build_report({"vanilla_tps_probe": 10.0}, _args())
    assert report["flops_per_emitted_token"] > 0


def test_validate_report_catches_missing_required_field() -> None:
    mod = _load_module()
    report = mod.build_report({"vanilla_tps_probe": 10.0}, _args())
    del report["mean"]  # required by schema
    errors = mod.validate_report(report, SCHEMA)
    assert any("mean" in e for e in errors)


def test_target_load_failure_fails_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load failure must surface as None (caller exits 3), not a placeholder."""
    import familydraft.targets.wrapper as wrapper_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("repo not reachable")

    monkeypatch.setattr(wrapper_mod.TargetModel, "load", staticmethod(_boom))
    mod = _load_module()
    measured = mod._acc_len("Qwen/Qwen3-8B", Path("runs/ckpt"), "structured", 8)
    assert measured is None
