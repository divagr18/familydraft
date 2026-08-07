"""F1 plan-compliance audit (locally-completable subset).

Checks the gate artifacts and pre-registration integrity the plan's F1 requires
(plan todo F1). NOT a full clean-clone audit (pod-deferred parts documented).

Checks:
  1. M1 gate tag v0.1-M1 exists.
  2. M2 gate report exists (docs/reports/oracle_report.md); v0.2-M2-NOGO implied
     by the report's NO-GO/revised history (no separate tag required).
  3. v0.3-M3 tag must NOT exist yet: M3 is not met (local verdict FAIL at 0.6B),
     so tagging it would be dishonest. The tag is created only when the pod 8B
     campaign PASSes.
  4. Verdict protocol predates the campaign (git ordering proof) - re-runs
     scripts/m3_order_check.py.
  5. Leak-proof assertion: sealed eval manifest integrity - re-runs
     scripts/verify_manifest.py (SHA-256 + item counts).
  6. Pre-registration immutability: verdict_protocol.yaml has no uncommitted
     working-tree modification at audit time.

Exit 0 only if every check passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []

    # 1. M1 tag
    rc, out = _run(["git", "tag", "-l", "v0.1-M1"], root)
    if rc != 0 or "v0.1-M1" not in out:
        failures.append("M1 tag v0.1-M1 missing")
    else:
        print("  OK  M1 gate tag v0.1-M1 present")

    # 2. M2 gate report
    if not (root / "docs" / "reports" / "oracle_report.md").exists():
        failures.append("M2 gate report docs/reports/oracle_report.md missing")
    else:
        print("  OK  M2 gate report present (v0.2-M2-NOGO implied by report)")

    # 3. v0.3-M3 tag must be absent (M3 not met yet)
    rc, out = _run(["git", "tag", "-l", "v0.3-M3"], root)
    if rc == 0 and "v0.3-M3" in out:
        failures.append("v0.3-M3 tag exists but M3 verdict is FAIL (tag premature)")
    else:
        print("  OK  v0.3-M3 tag absent (M3 FAIL - tag correctly withheld)")

    # 4. Verdict protocol predates campaign (git ordering proof)
    rc, out = _run([sys.executable, "scripts/m3_order_check.py"], root)
    if rc != 0:
        failures.append(f"m3_order_check failed: {out[-300:]}")
    else:
        print("  OK  verdict protocol predates campaign (m3_order_check PASS)")

    # 5. Leak-proof: sealed eval manifest integrity
    rc, out = _run([sys.executable, "scripts/verify_manifest.py"], root)
    if rc != 0:
        failures.append(f"verify_manifest failed: {out[-300:]}")
    else:
        print("  OK  sealed eval manifest integrity (leak-proof re-run PASS)")

    # 6. Protocol immutability at audit time (no working-tree edit)
    rc, out = _run(["git", "diff", "--quiet", "--", "configs/verdict_protocol.yaml"], root)
    if rc != 0:
        failures.append("configs/verdict_protocol.yaml has uncommitted working-tree changes")
    else:
        print("  OK  verdict_protocol.yaml clean in working tree at audit time")

    if failures:
        print("\nF1 AUDIT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nF1 AUDIT (local subset): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
