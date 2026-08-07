"""Phase-1 campaign runner (M3 / plan todo 22).

Aggregates the per-system/per-task baseline reports (`runs/baselines/*.json`
produced by scripts/run_baselines.py) into `runs/results/phase1.csv` with the
columns m3_verdict.py expects. Also validates every report against
configs/baseline_report.schema.json.

Row coverage check: asserts all (system x task_class) rows from the verdict
protocol are present; missing rows are reported and cause a non-zero exit so a
partial campaign cannot silently produce a verdict.

Usage: uv run python scripts/run_phase1.py [--schema configs/baseline_report.schema.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROTOCOL = Path("configs/verdict_protocol.yaml")
BASELINES_DIR = Path("runs/baselines")
OUT_CSV = Path("runs/results/phase1.csv")

CSV_COLUMNS = [
    "system",
    "task_class",
    "repo",
    "target_tokens_per_second",
    "std",
    "runs",
    "config_hash",
    "flops_per_emitted_token",
    "dense_equivalent_layers",
    "exactness_fp32_exact",
    "exactness_bf16_exact_match_rate",
    "tokens_per_round",
]


def _validate(report: dict, schema_path: Path) -> list[str]:
    """Best-effort schema validation without a jsonschema dependency."""
    errors = []
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["schema file unreadable"]
    required = schema.get("required", [])
    for field in required:
        if field not in report:
            errors.append(f"missing required field {field!r}")
    if "system" in report:
        enum = schema.get("properties", {}).get("system", {}).get("enum", [])
        if report["system"] not in enum:
            errors.append(f"system {report['system']!r} not in enum")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="configs/baseline_report.schema.json")
    args = parser.parse_args()
    schema_path = Path(args.schema)

    if not PROTOCOL.exists():
        print(f"run_phase1: missing protocol {PROTOCOL}", file=sys.stderr)
        return 2
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    systems = protocol["systems"]
    task_classes = protocol["task_classes"]
    # EAGLE-3 requires SpecForge training on the pod (todo 13); the verdict
    # protocol records it as a reported gap, so a missing eagle3 row does not
    # block a locally-run verdict.
    optional_systems: set[str] = set()
    if protocol.get("verdict", {}).get("eagle3_reported_as_gap"):
        optional_systems.add("eagle3_specforge")

    if not BASELINES_DIR.exists():
        print(f"run_phase1: no baselines dir {BASELINES_DIR}", file=sys.stderr)
        return 2

    reports = []
    for path in sorted(BASELINES_DIR.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"run_phase1: unreadable report {path.name}", file=sys.stderr)
            return 2
        errors = _validate(report, schema_path)
        if errors:
            print(f"run_phase1: schema violations in {path.name}: {errors}", file=sys.stderr)
            return 2
        reports.append(report)

    present = {(r["system"], r["task_class"]) for r in reports}
    missing = [
        (s, tc)
        for s in systems
        for tc in task_classes
        if (s, tc) not in present and s not in optional_systems
    ]
    gaps = [
        (s, tc)
        for s in systems
        for tc in task_classes
        if (s, tc) not in present and s in optional_systems
    ]
    if gaps:
        print("run_phase1: optional rows absent (recorded as reported gaps):")
        for s, tc in sorted(gaps):
            print(f"  - {s} / {tc}")
    if missing:
        print("run_phase1: MISSING rows (campaign incomplete):")
        for s, tc in sorted(missing):
            print(f"  - {s} / {tc}")
        print("run_phase1: not writing phase1.csv (partial campaign cannot yield a verdict)")
        return 78

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in sorted(reports, key=lambda x: (x["system"], x["task_class"])):
            writer.writerow({
                "system": r["system"],
                "task_class": r["task_class"],
                "repo": r.get("repo", ""),
                "target_tokens_per_second": r["target_tokens_per_second"],
                "std": r.get("std", 0.0),
                "runs": r.get("runs", 0),
                "config_hash": r.get("config_hash", ""),
                "flops_per_emitted_token": r.get("flops_per_emitted_token", ""),
                "dense_equivalent_layers": r.get("dense_equivalent_layers", ""),
                "exactness_fp32_exact": r.get("exactness", {}).get("fp32_exact", ""),
                "exactness_bf16_exact_match_rate": r.get("exactness", {}).get(
                    "bf16_exact_match_rate", ""
                ),
                "tokens_per_round": r.get("tokens_per_round", ""),
            })
    print(f"run_phase1: wrote {OUT_CSV} with {len(reports)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
