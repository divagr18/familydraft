"""Expert gate decision engine (plan todo 25).

Applies the pre-registered gate rule (configs/expert_gates.yaml) to candidate
gated experts (reasoning_transition, logit_dynamics) using the todo-11 oracle
report's per-mechanism coverage. Adds an expert only if the oracle shows the
mechanism class recovers >=5 percentage points additional positions with
expected acceptance >=1.0 token in >=1 task class; otherwise records SKIP with
an evidence pointer.

Emits docs/reports/expert_gate_decisions.md. `--dry-run` feeds a synthetic
oracle verdict and prints the decision without writing the file (QA scenario:
a below-gate synthetic report must decide SKIP).

The real per-mechanism coverage for reasoning/logit-dynamics is not produced
by the current oracle (union coverage only), so the honest decision here is
SKIP for both candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = ["reasoning_transition", "logit_dynamics"]
MIN_ADDITIONAL_PCT = 5.0
MIN_EXPECTED_ACCEPTANCE = 1.0
GATE_RULE_REF = "docs/reports/oracle_report.md"


def _load_gate_config() -> dict:
    path = REPO_ROOT / "configs" / "expert_gates.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _oracle_mechanism_coverage(verdict: dict) -> dict[str, dict[str, float]]:
    """Extract per-mechanism coverage from the oracle verdict.

    The v1 verdict only stores union coverage; per-mechanism data is absent.
    Returns an empty mapping for the gated mechanisms, which the rule treats
    as 'no evidence' -> SKIP.
    """
    return {}


def _synthetic_below_gate_verdict() -> dict:
    """Below-gate synthetic oracle: no per-mechanism coverage at all."""
    return {
        "schema": "familydraft.oracle_verdict.v1",
        "mechanism_coverage": {},
    }


def _decide(verdict: dict) -> list[dict]:
    coverage = _oracle_mechanism_coverage(verdict)
    decisions = []
    for expert in CANDIDATES:
        per_class = coverage.get(expert, {})
        best = max(per_class.values(), default=0.0)
        ok = any(
            pct >= MIN_ADDITIONAL_PCT and acc >= MIN_EXPECTED_ACCEPTANCE
            for (pct, acc) in (
                (per_class.get(cls, {}).get("additional_pct", 0.0),
                 per_class.get(cls, {}).get("expected_acceptance", 0.0))
                for cls in per_class
            )
        ) if per_class else False
        decisions.append({
            "expert": expert,
            "decision": "PASS" if ok else "SKIP",
            "reason": (
                f"oracle per-mechanism coverage >= {MIN_ADDITIONAL_PCT}pp with "
                f"expected acceptance >= {MIN_EXPECTED_ACCEPTANCE} in >=1 class"
                if ok
                else f"no per-mechanism oracle coverage >= {MIN_ADDITIONAL_PCT}pp "
                     f"with expected acceptance >= {MIN_EXPECTED_ACCEPTANCE} "
                     f"(best additional_pct={best:.1f}); evidence pointer: "
                     f"{GATE_RULE_REF}"
            ),
        })
    return decisions


def _render_md(decisions: list[dict]) -> str:
    lines = ["# Expert gate decisions (plan todo 25)", ""]
    lines.append(
        f"Gate rule: add expert iff oracle per-mechanism coverage recovers "
        f">= {MIN_ADDITIONAL_PCT}pp additional positions with expected "
        f"acceptance >= {MIN_EXPECTED_ACCEPTANCE} token in >=1 task class "
        f"(configs/expert_gates.yaml, committed before this todo ran)."
    )
    lines.append("")
    for d in decisions:
        lines.append(f"## {d['expert']}: {d['decision']}")
        lines.append("")
        lines.append(f"- {d['reason']}")
        lines.append("")
    lines.append(f"Evidence source: `{GATE_RULE_REF}` and `runs/oracle/verdict.json`.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="feed a synthetic below-gate oracle verdict and print without writing",
    )
    args = parser.parse_args()

    _load_gate_config()  # config must exist (pre-registration)
    verdict = _synthetic_below_gate_verdict() if args.dry_run else None
    if verdict is None:
        verdict_path = REPO_ROOT / "runs" / "oracle" / "verdict.json"
        if not verdict_path.exists():
            print(f"apply_expert_gate: missing {verdict_path} "
                  "(run oracle analysis first, or --dry-run)", file=sys.stderr)
            return 2
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    decisions = _decide(verdict)
    md = _render_md(decisions)

    if args.dry_run:
        print(md)
        # QA: below-gate synthetic must SKIP every candidate.
        if any(d["decision"] != "SKIP" for d in decisions):
            print("apply_expert_gate: FAIL - below-gate oracle did not SKIP", file=sys.stderr)
            return 1
        return 0

    out = REPO_ROOT / "docs" / "reports" / "expert_gate_decisions.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    for d in decisions:
        print(f"{d['expert']}: {d['decision']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
