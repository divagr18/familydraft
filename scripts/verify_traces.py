"""Verify trace campaign manifest integrity (plan todo 10).

Accepts a traces dir and verifies every shard listed in MANIFEST.traces.json
exists with a matching sha256. Exit 0 on pass; exit 1 naming every corrupt or
missing shard (QA failure path: a single corrupted byte must be caught and
named). Partial shards are detected via a length/schema check on each line.

Usage: python scripts/verify_traces.py /workspace/traces
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _line_is_valid(line: str) -> bool:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(obj, dict)
        and "chosen_token" in obj
        and "topk" in obj
        and isinstance(obj["topk"], dict)
    )


def _verify_shard(path: Path) -> list[str]:
    """Return list of problems for one shard (empty = clean)."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"unreadable: {exc}"]
    nonempty = [ln for ln in text.splitlines() if ln.strip()]
    if not nonempty:
        return ["empty shard"]
    bad = [i for i, ln in enumerate(nonempty) if not _line_is_valid(ln)]
    if bad:
        problems.append(f"invalid JSONL line(s): {bad[:5]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces_dir", help="path to /workspace/traces")
    args = parser.parse_args()

    root = Path(args.traces_dir)
    manifest_path = root / "MANIFEST.traces.json"
    if not manifest_path.exists():
        print(f"verify_traces: FAIL - {manifest_path} missing", file=sys.stderr)
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"verify_traces: FAIL - manifest unreadable: {exc}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for key, entry in manifest.items():
        shard = root / key
        if not shard.exists():
            failures.append(f"missing shard: {key}")
            continue
        expected = entry.get("sha256")
        actual = _sha256(shard)
        if expected is not None and actual != expected:
            failures.append(f"sha256 mismatch: {key} (expected {expected}, got {actual})")
        for problem in _verify_shard(shard):
            failures.append(f"{key}: {problem}")

    if failures:
        print(f"verify_traces: FAIL - {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"verify_traces: PASS - {len(manifest)} shards verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
