"""Serving matrix + equivalence checker tests (plan todo 28, locally-completable).

Acceptance criteria (plan todo 28):
  - runs/phase4/serving_matrix.json validates against the pre-registered
    configs/serving_matrix.schema.json (3 batch sizes x 3 systems x
    throughput + latency; sglang_version + config_hash recorded)
  - a schema-valid artifact REQUIRES all 9 cells with positive values
    (schema exclusiveMinimum 0) - incomplete/zero data fails honestly
  - the structural dry-run preview is NOT schema-valid (guard asserted)
  - equivalence check: 50/50 identical greedy outputs (asserted)
  - QA failure scenario: seed a deliberate divergence -> equivalence check
    fails NAMING divergent prompt ids
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parent.parent
SCHEMA = _REPO / "configs" / "serving_matrix.schema.json"


def _load_matrix_mod():
    spec = importlib.util.spec_from_file_location(
        "run_serving_matrix", _REPO / "scripts" / "run_serving_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_equiv_mod():
    spec = importlib.util.spec_from_file_location(
        "check_serving_equivalence", _REPO / "scripts" / "check_serving_equivalence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _full_measured() -> dict:
    """Complete 9-cell measured dict with positive values."""
    out = {}
    for system in ("familydraft_dag", "vanilla_ar", "eagle3_specforge"):
        for batch in (1, 8, 32):
            out[(system, batch)] = {
                "throughput_tokens_per_sec": 20.0,
                "latency_ms_per_request": 50.0,
            }
    return out


def _sample_outputs(n: int = 50) -> dict[str, list[int]]:
    return {str(i): [i % 7 + 10, 3, 9] for i in range(n)}


def _write_jsonl(path: Path, outputs: dict[str, list[int]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for pid, tokens in outputs.items():
            fh.write(json.dumps({"id": pid, "tokens": tokens}) + "\n")


def test_schema_requirements() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert "systems" in schema["required"]
    assert "batch_sizes" in schema["required"]
    assert "cells" in schema["required"]
    # plan: 3 systems incl. the three required by the batch matrix
    assert schema["properties"]["systems"]["minItems"] == 3


def test_build_matrix_validates_against_schema() -> None:
    mod = _load_matrix_mod()
    matrix = mod.build_matrix("v0.4.1", "a" * 16, _full_measured())
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(matrix, schema)
    assert len(matrix["cells"]) == 3 * 3  # 3 systems x 3 batch sizes
    assert set(mod.SYSTEMS) == {"familydraft_dag", "vanilla_ar", "eagle3_specforge"}
    assert mod.BATCH_SIZES == [1, 8, 32]


def test_build_matrix_accepts_measured_cells() -> None:
    mod = _load_matrix_mod()
    measured = _full_measured()
    measured[("familydraft_dag", 1)] = {
        "throughput_tokens_per_sec": 12.5, "latency_ms_per_request": 80.0}
    matrix = mod.build_matrix("v0.4.1", "a" * 16, measured)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(matrix, schema)
    dag_cell = [c for c in matrix["cells"]
                if c["system"] == "familydraft_dag" and c["batch_size"] == 1][0]
    assert dag_cell["throughput_tokens_per_sec"] == 12.5


def test_incomplete_measured_data_fails_honestly() -> None:
    """A schema-valid artifact requires ALL 9 cells; a missing cell raises."""
    mod = _load_matrix_mod()
    measured = _full_measured()
    del measured[("eagle3_specforge", 32)]
    with pytest.raises(ValueError, match="missing measured cell"):
        mod.build_matrix("v0.4.1", "a" * 16, measured)


def test_non_positive_cell_fails_honestly() -> None:
    """Zero cells violate schema exclusiveMinimum 0 and must raise."""
    mod = _load_matrix_mod()
    measured = _full_measured()
    measured[("vanilla_ar", 1)] = {
        "throughput_tokens_per_sec": 0.0, "latency_ms_per_request": 50.0}
    with pytest.raises(ValueError, match="positive"):
        mod.build_matrix("v0.4.1", "a" * 16, measured)


def test_structural_preview_is_not_schema_valid() -> None:
    """The structural dry-run preview (zeros) must NOT validate - the schema
    guard exclusiveMinimum 0 exists precisely to forbid placeholder artifacts."""
    mod = _load_matrix_mod()
    preview = mod.build_structural_preview("v0.4.1", "a" * 16)
    errors = mod._validate(preview, SCHEMA)
    assert errors, "structural preview must not validate against the schema"


def test_equivalence_pass_on_identical_outputs(tmp_path: Path) -> None:
    mod = _load_equiv_mod()
    ref = _sample_outputs(50)
    cand = {k: list(v) for k, v in ref.items()}  # identical copy
    ref_p = tmp_path / "ref.jsonl"
    cand_p = tmp_path / "cand.jsonl"
    _write_jsonl(ref_p, ref)
    _write_jsonl(cand_p, cand)
    assert mod.compare(mod.load_outputs(ref_p), mod.load_outputs(cand_p)) == []


def test_equivalence_fails_naming_divergent_prompts(tmp_path: Path) -> None:
    mod = _load_equiv_mod()
    ref = _sample_outputs(50)
    cand = {k: list(v) for k, v in ref.items()}
    cand["17"] = [99, 99]  # deliberate divergence
    ref_p = tmp_path / "ref.jsonl"
    cand_p = tmp_path / "cand.jsonl"
    _write_jsonl(ref_p, ref)
    _write_jsonl(cand_p, cand)
    divergent = mod.compare(mod.load_outputs(ref_p), mod.load_outputs(cand_p))
    assert "17" in divergent  # names the divergent prompt id


def test_cli_seed_divergence_fails(tmp_path: Path) -> None:
    ref = _sample_outputs(50)
    cand = {k: list(v) for k, v in ref.items()}
    ref_p = tmp_path / "ref.jsonl"
    cand_p = tmp_path / "cand.jsonl"
    _write_jsonl(ref_p, ref)
    _write_jsonl(cand_p, cand)
    proc = subprocess.run(
        [sys.executable, "scripts/check_serving_equivalence.py",
         "--reference", str(ref_p), "--candidate", str(cand_p),
         "--seed-divergence"],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "FAIL" in proc.stderr
    assert "0" in proc.stderr  # seeded at the first (0) prompt
