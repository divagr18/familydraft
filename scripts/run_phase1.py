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


def _reproduce_row(report: dict, tolerance_pct: float) -> int:
    """Re-run one baseline row from its recorded run_args and compare the
    measured tokens/sec against the stored value (plan todo 22 acceptance:
    spot-check rows within ±3%)."""
    import subprocess

    run_args = report.get("run_args")
    if not run_args:
        print(f"run_phase1: row {report.get('system')}/{report.get('task_class')} "
              "has no run_args recorded (old report); regenerate it", file=sys.stderr)
        return 2

    cmd = [
        sys.executable, "scripts/run_baselines.py",
        "--system", run_args["system"],
        "--task-class", run_args["task_class"],
        "--repo", run_args["repo"],
        "--max-new", str(run_args["max_new"]),
        "--spec-len", str(run_args["spec_len"]),
        "--max-prompts", str(run_args["max_prompts"]),
        "--runs", str(run_args["runs"]),
        "--out", str(BASELINES_DIR / f"repro_{run_args['system']}_{run_args['task_class']}.json"),
    ]
    if run_args.get("general_checkpoint"):
        cmd += ["--general-checkpoint", run_args["general_checkpoint"]]
    if run_args.get("router_weights"):
        cmd += ["--router-weights", run_args["router_weights"]]
    if run_args.get("ablation"):
        cmd += ["--ablation", run_args["ablation"]]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"run_phase1: re-run of {run_args['system']}/{run_args['task_class']} "
              f"failed (exit {proc.returncode}): {proc.stderr[-500:]}", file=sys.stderr)
        return 1

    repro_path = BASELINES_DIR / f"repro_{run_args['system']}_{run_args['task_class']}.json"
    try:
        repro = json.loads(repro_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"run_phase1: re-run report unreadable: {repro_path}", file=sys.stderr)
        return 1

    orig = float(report["target_tokens_per_second"])
    new = float(repro["target_tokens_per_second"])
    pct = 100.0 * abs(new - orig) / max(1e-9, orig)
    status = "OK" if pct <= tolerance_pct else "DIVERGED"
    print(f"run_phase1: row {run_args['system']}/{run_args['task_class']} hash "
          f"{report.get('config_hash')}: orig={orig:.2f} tps re-run={new:.2f} tps "
          f"delta={pct:.2f}% ({status})")
    return 0 if status == "OK" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="configs/baseline_report.schema.json")
    parser.add_argument(
        "--row",
        default="",
        help="config_hash of a baseline row to reproduce (re-run + ±3% tps check)",
    )
    parser.add_argument("--tolerance-pct", type=float, default=3.0)
    args = parser.parse_args()
    schema_path = Path(args.schema)

    if args.row:
        if not BASELINES_DIR.exists():
            print(f"run_phase1: no baselines dir {BASELINES_DIR}", file=sys.stderr)
            return 2
        for path in sorted(BASELINES_DIR.glob("*.json")):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if report.get("config_hash") == args.row:
                return _reproduce_row(report, args.tolerance_pct)
        print(f"run_phase1: no row with config_hash {args.row}", file=sys.stderr)
        return 2

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
        if path.stem.startswith("repro_"):
            continue  # reproducibility spot-check artifacts, not campaign rows
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
