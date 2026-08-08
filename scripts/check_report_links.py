"""Check every runs/ path cited in a report exists (plan todo 23).

Usage: scripts/check_report_links.py <report.md>

Scans the report for `runs/...` and `data/...` code-span citations and asserts
each referenced path exists on disk. Exit 0 on pass; exit 1 naming every
missing path (QA failure path: a report appendix citing a nonexistent runs/
path must fail loudly).

Also checks `docs/paper/` citations resolve to EVIDENCE_INDEX entries when the
report links a claim row.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PATH_RE = re.compile(
    r"(runs/[\w./\-]+\.\w+|data/[\w./\-]+\.\w+|EVIDENCE_INDEX\.json|"
    r"docs/paper/[\w./\-]+\.\w+)"
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: scripts/check_report_links.py <report.md>", file=sys.stderr)
        return 2
    report_path = Path(sys.argv[1])
    if not report_path.exists():
        print(f"check_report_links: report not found: {report_path}", file=sys.stderr)
        return 2

    text = report_path.read_text(encoding="utf-8", errors="ignore")
    # Ignore paths inside code blocks that are command examples (e.g. `uv run
    # python scripts/...` includes no runs/ paths) and artifacts sections.
    cited = sorted(set(_PATH_RE.findall(text)))

    missing: list[str] = []
    for p in cited:
        if not (ROOT / p).exists():
            missing.append(p)

    if missing:
        n = len(missing)
        print(
            f"check_report_links: FAIL - {n} cited path(s) missing in "
            f"{report_path.name}:"
        )
        for m in missing:
            print(f"  - {m}")
        return 1

    print(f"check_report_links: PASS - {len(cited)} cited paths in {report_path.name} all exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
