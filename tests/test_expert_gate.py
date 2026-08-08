"""Expert gate decisions tests (plan todo 25, locally-completable portion).

Acceptance criteria:
  - gate decision file docs/reports/expert_gate_decisions.md exists with a
    per-expert decision + evidence pointer into oracle_report.md
  - for SKIP experts, no implementation code exists (file-absence assertion)
  - --dry-run with a synthetic below-gate oracle decides SKIP (QA logic check)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

CANDIDATES = ["reasoning_transition", "logit_dynamics"]


def test_gate_decisions_file_exists() -> None:
    path = _REPO / "docs" / "reports" / "expert_gate_decisions.md"
    assert path.exists(), "expert_gate_decisions.md missing"
    text = path.read_text(encoding="utf-8")
    assert "oracle_report.md" in text, "decision file must cite oracle_report.md"
    for expert in CANDIDATES:
        assert f"{expert}:" in text, f"decision file missing entry for {expert}"


def test_skipped_experts_have_no_implementation() -> None:
    """File-absence assertion: SKIP experts must not have implementation code."""
    src = _REPO / "src" / "familydraft" / "experts"
    for expert in CANDIDATES:
        # map expert name to a plausible module filename
        module = expert.replace("_", "")
        for candidate in (module + ".py", expert + ".py"):
            assert not (src / candidate).exists(), (
                f"SKIP expert {expert} must have no implementation, found {candidate}"
            )


def test_dry_run_below_gate_decides_skip() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/apply_expert_gate.py", "--dry-run"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SKIP" in proc.stdout
    assert "PASS" not in proc.stdout
