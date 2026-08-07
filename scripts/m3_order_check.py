"""M3 pre-registration integrity check (plan todo 22 gate guard).

Asserts configs/verdict_protocol.yaml is committed in git history and predates
the first Phase-1 campaign run (runs/results/phase1.csv): the verdict protocol
must have existed before any campaign row was measured, so the thresholds could
not have been tuned post-hoc to fit the results.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROTOCOL = "configs/verdict_protocol.yaml"
CSV = "runs/results/phase1.csv"


def _commit_ts(path: str) -> tuple[str, int] | None:
    log = subprocess.run(
        ["git", "log", "--follow", "--format=%H %ct", "--", path],
        capture_output=True,
        text=True,
    )
    if log.returncode != 0 or not log.stdout.strip():
        return None
    first_line = log.stdout.strip().splitlines()[0]
    parts = first_line.split()
    return parts[0], int(parts[1])


def main() -> int:
    proto = _commit_ts(PROTOCOL)
    if proto is None:
        print(f"m3_order_check: {PROTOCOL} has no git history (not committed)", file=sys.stderr)
        return 1
    proto_sha, proto_ts = proto

    if not Path(CSV).exists():
        print(f"m3_order_check: {CSV} not found; run scripts/run_phase1.py first", file=sys.stderr)
        return 1
    csv_ts = int(Path(CSV).stat().st_mtime)

    if proto_ts >= csv_ts:
        print(
            f"m3_order_check: {PROTOCOL} committed {proto_ts} >= phase1.csv mtime {csv_ts}; "
            "protocol did not predate the campaign",
            file=sys.stderr,
        )
        return 1
    print(
        f"m3_order_check: PASS ({PROTOCOL} @ {proto_sha} committed before phase1.csv)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
