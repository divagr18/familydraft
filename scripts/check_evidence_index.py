"""Check EVIDENCE_INDEX.json resolves every paper claim (plan todo 30).

Acceptance criteria: every claim row in docs/paper/claims.csv must reference an
evidence path that exists in EVIDENCE_INDEX.json. Exit 0 on pass; on failure
exit 1 naming every claim row whose evidence is missing or unindexed (QA
failure path: a claim citing a bogus evidence id must fail loudly).

Also asserts the index itself is fresh: git_sha in EVIDENCE_INDEX.json equals
the current HEAD (stale index = unverifiable evidence).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "EVIDENCE_INDEX.json"
CLAIMS = ROOT / "docs" / "paper" / "claims.csv"


def main() -> int:
    failures: list[str] = []

    if not INDEX.exists():
        print("check_evidence_index: EVIDENCE_INDEX.json missing "
              "(run scripts/build_evidence_index.py)", file=sys.stderr)
        return 1
    if not CLAIMS.exists():
        print("check_evidence_index: docs/paper/claims.csv missing", file=sys.stderr)
        return 1

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    indexed_paths = {e["path"] for e in index.get("entries", [])}

    # Index freshness: git_sha must match HEAD.
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    head = proc.stdout.strip() if proc.returncode == 0 else "unknown"
    if index.get("git_sha") != head:
        failures.append(
            f"EVIDENCE_INDEX.json git_sha {index.get('git_sha')} != HEAD {head}; "
            "index is stale - rebuild with scripts/build_evidence_index.py"
        )

    # Every claim must resolve.
    bad_rows: list[str] = []
    with CLAIMS.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            claim_id = row.get("claim_id", "?")
            ev = (row.get("evidence_path") or "").replace("\\", "/")
            if not ev:
                bad_rows.append(f"{claim_id}: empty evidence_path")
                continue
            if ev not in indexed_paths:
                bad_rows.append(f"{claim_id}: evidence '{ev}' not in EVIDENCE_INDEX.json")

    if bad_rows:
        failures.append(f"claims with missing/unindexed evidence ({len(bad_rows)}):")
        for r in bad_rows:
            failures.append(f"  - {r}")

    if failures:
        print("check_evidence_index: FAIL")
        for f in failures:
            print(f"  {f}")
        return 1

    print(f"check_evidence_index: PASS - {len(indexed_paths)} indexed artifacts, "
          f"all claims resolve (git_sha {index.get('git_sha')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
