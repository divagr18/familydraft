"""Build EVIDENCE_INDEX.json (plan todo 30).

Indexes every run artifact, trace manifest, checkpoint reference and report
with sha256 + git sha + config hash so every claim in docs/paper/claims.csv
resolves to a verifiable artifact. Exit 0 on success.

Indexed roots:
  - runs/            all files (baseline reports, logs, checkpoints, P1 probes)
  - docs/reports/    verdict + oracle reports
  - data/eval/       sealed eval manifest + datasets (item counts, not contents)

Each entry: { path, sha256, git_sha, size_bytes, kind } where kind is inferred
from the path (baseline_report / log / checkpoint / probe / report / manifest).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "EVIDENCE_INDEX.json"

INDEX_ROOTS = ("runs", "docs/reports")
MANIFEST_PATH = ROOT / "data" / "eval" / "MANIFEST.json"


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _kind(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith("runs/baselines/"):
        return "baseline_report" if "abl_" not in name else "ablation_report"
    if rel.startswith("runs/results/"):
        return "campaign_result"
    if rel.endswith(".pt"):
        return "checkpoint"
    if rel.startswith("runs/trainlogs/"):
        return "training_log"
    if rel.startswith("docs/reports/"):
        return "report"
    if rel.endswith(".json"):
        return "json_artifact"
    if rel.endswith(".log"):
        return "log"
    if rel.endswith(".py"):
        return "probe_script"
    return "artifact"


def main() -> int:
    if not ROOT.joinpath(".git").exists():
        print("build_evidence_index: not a git repo", file=sys.stderr)
        return 2

    git_sha = _git_sha()
    entries: list[dict] = []

    for root_rel in INDEX_ROOTS:
        root = ROOT / root_rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            if path.name == ".gitkeep":
                continue
            entries.append({
                "path": rel,
                "sha256": _sha256(path),
                "git_sha": git_sha,
                "size_bytes": path.stat().st_size,
                "kind": _kind(rel),
            })

    # Sealed eval manifest + dataset item counts (not full contents).
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entries.append({
            "path": "data/eval/MANIFEST.json",
            "sha256": _sha256(MANIFEST_PATH),
            "git_sha": git_sha,
            "size_bytes": MANIFEST_PATH.stat().st_size,
            "kind": "manifest",
            "total_prompts": manifest.get("round_trip_validation", {}).get("total_prompts"),
        })
        for ds_name, ds_info in manifest.get("datasets", {}).items():
            df = ROOT / "data" / "eval" / ds_info["data_file"]
            if df.exists():
                entries.append({
                    "path": f"data/eval/{ds_info['data_file']}",
                    "sha256": _sha256(df),
                    "git_sha": git_sha,
                    "size_bytes": df.stat().st_size,
                    "kind": "eval_dataset",
                    "item_count": ds_info.get("item_count"),
                })

    index = {
        "schema": "familydraft.evidence_index.v1",
        "git_sha": git_sha,
        "built_at": sys.version_info[:2],
        "entries": entries,
    }
    OUT.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"build_evidence_index: {len(entries)} entries -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
