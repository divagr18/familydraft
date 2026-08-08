"""Phase-4 serving batch matrix producer (plan todo 28, locally-completable).

Emits runs/phase4/serving_matrix.json validated against the pre-registered
configs/serving_matrix.schema.json (3 batch sizes x 3 systems x
throughput + latency, plus sglang_version and config_hash).

A schema-valid artifact REQUIRES all 9 (system x batch_size) cells measured
with positive throughput/latency (schema exclusiveMinimum: 0). Incomplete or
non-positive measured data fails honestly (no placeholder-zero artifact - the
schema guard is the point, asserted in tests). The equivalence check (50/50
identical greedy outputs) is separately enforced by
scripts/check_serving_equivalence.py.

`--dry-run` writes a STRUCTURAL PREVIEW (zeros + note) for inspecting the
shape before the pod serves real numbers. It is NOT schema-valid by design
and must not be treated as an artifact (asserted in tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "configs" / "serving_matrix.schema.json"

SYSTEMS = ["familydraft_dag", "vanilla_ar", "eagle3_specforge"]
BATCH_SIZES = [1, 8, 32]


def _validate(matrix: dict, schema_path: Path) -> list[str]:
    import jsonschema

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["schema file unreadable"]
    try:
        jsonschema.validate(matrix, schema)
        return []
    except jsonschema.ValidationError as exc:
        return [exc.message]


def build_matrix(sglang_version: str, config_hash: str,
                 measured: dict) -> dict:
    """Assemble a schema-valid matrix from a COMPLETE measured dict.

    `measured` maps (system, batch_size) -> {"throughput_tokens_per_sec",
    "latency_ms_per_request"} and must cover every one of the 9 cells with
    positive values; anything less raises ValueError (honest failure - the
    schema forbids zero/placeholder cells).
    """
    cells = []
    for system in SYSTEMS:
        for batch_size in BATCH_SIZES:
            key = (system, batch_size)
            if key not in measured:
                raise ValueError(
                    f"missing measured cell for {system}/batch {batch_size} "
                    f"(all {len(SYSTEMS) * len(BATCH_SIZES)} cells required)")
            cell = measured[key]
            if cell.get("throughput_tokens_per_sec", 0) <= 0 or \
                    cell.get("latency_ms_per_request", 0) <= 0:
                raise ValueError(
                    f"cell {system}/batch {batch_size} must have positive "
                    f"throughput and latency (schema exclusiveMinimum 0)")
            cells.append({
                "system": system,
                "batch_size": batch_size,
                "throughput_tokens_per_sec": cell["throughput_tokens_per_sec"],
                "latency_ms_per_request": cell["latency_ms_per_request"],
            })
    return {
        "schema": "familydraft.serving_matrix.v1",
        "sglang_version": sglang_version,
        "config_hash": config_hash,
        "systems": SYSTEMS,
        "batch_sizes": BATCH_SIZES,
        "cells": cells,
    }


def build_structural_preview(sglang_version: str, config_hash: str) -> dict:
    """Structural preview: zeros + note per cell. NOT schema-valid by design
    (the schema forbids 0 throughput); for shape inspection only."""
    cells = []
    for system in SYSTEMS:
        for batch_size in BATCH_SIZES:
            cells.append({
                "system": system,
                "batch_size": batch_size,
                "throughput_tokens_per_sec": 0.0,
                "latency_ms_per_request": 0.0,
                "note": "structural preview (pod serves real numbers)",
            })
    return {
        "schema": "familydraft.serving_matrix.v1",
        "sglang_version": sglang_version,
        "config_hash": config_hash,
        "systems": SYSTEMS,
        "batch_sizes": BATCH_SIZES,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-version", default="unknown",
                        help="pinned SGLang version (pod records the real pin)")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--measured", default="",
                        help="JSON file: {system: {batch_size: {throughput_tokens_per_sec, "
                             "latency_ms_per_request}}} - ALL 9 cells required")
    parser.add_argument("--dry-run", action="store_true",
                        help="write a structural preview (NOT schema-valid)")
    parser.add_argument("--out", default="runs/phase4/serving_matrix.json")
    args = parser.parse_args()

    config_hash = args.config_hash or hashlib.sha256(
        json.dumps({"systems": SYSTEMS, "batch_sizes": BATCH_SIZES},
                   sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    if args.dry_run:
        matrix = build_structural_preview(args.sglang_version, config_hash)
        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        print(f"run_serving_matrix: wrote STRUCTURAL PREVIEW (NOT schema-valid) "
              f"to {out_path.relative_to(REPO_ROOT)} - do not treat as an artifact")
        return 0

    if not args.measured:
        print("run_serving_matrix: --measured FILE required for a schema-valid "
              "artifact (or --dry-run for a structural preview)", file=sys.stderr)
        return 2

    try:
        raw = json.loads(Path(args.measured).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"run_serving_matrix: measured file unreadable: {exc}", file=sys.stderr)
        return 2

    measured: dict = {}
    for system, by_batch in raw.items():
        for batch_str, cell in by_batch.items():
            measured[(system, int(batch_str))] = cell

    try:
        matrix = build_matrix(args.sglang_version, config_hash, measured)
    except ValueError as exc:
        print(f"run_serving_matrix: {exc}", file=sys.stderr)
        return 3

    errors = _validate(matrix, SCHEMA_PATH)
    if errors:
        print(f"run_serving_matrix: matrix failed schema validation: {errors}",
              file=sys.stderr)
        return 1

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"run_serving_matrix: wrote {out_path.relative_to(REPO_ROOT)} "
          f"({len(matrix['cells'])} cells, schema-valid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
