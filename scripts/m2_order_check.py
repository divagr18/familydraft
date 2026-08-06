"""M2 pre-registration integrity check (plan todo 11 gate guard).

Asserts configs/oracle_thresholds.yaml is committed in git history and has
not been modified since its first commit — proof the gate thresholds were
pre-registered before any analysis run and never tuned post-hoc.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    config = "configs/oracle_thresholds.yaml"
    log = subprocess.run(
        ["git", "log", "--follow", "--format=%H %ct", "--", config],
        capture_output=True,
        text=True,
    )
    if log.returncode != 0 or not log.stdout.strip():
        print(f"m2_order_check: {config} has no git history (not committed)", file=sys.stderr)
        return 1
    first_sha = log.stdout.strip().splitlines()[-1].split()[0]
    diff = subprocess.run(
        ["git", "diff", "--quiet", first_sha, "HEAD", "--", config]
    )
    if diff.returncode != 0:
        print(
            f"m2_order_check: {config} changed since first commit {first_sha}; "
            "pre-registration violated",
            file=sys.stderr,
        )
        return 1
    print(f"m2_order_check: PASS ({config} unchanged since {first_sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
