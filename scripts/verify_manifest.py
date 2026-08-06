"""Verify data/eval/MANIFEST.json integrity.

Re-hashes all listed data files and checks item counts.
Exit 0 on pass; exit 1 listing offending files on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EVAL_ROOT = Path("data/eval")
MANIFEST_PATH = EVAL_ROOT / "MANIFEST.json"

# Pinned expected counts (MBPP is discovered at build; stored in manifest)
PINNED_COUNTS = {
    "mtbench": 80,
    "humaneval": 164,
    "gsm8k": 1319,
    "structured": 100,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl_items(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"FAIL: {MANIFEST_PATH} not found", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})

    offending: list[str] = []

    for ds_name, ds_info in datasets.items():
        data_file = ds_info.get("data_file")
        expected_hash = ds_info.get("sha256")
        expected_count = ds_info.get("item_count")

        if not data_file:
            offending.append(f"{ds_name}: missing data_file in manifest")
            continue

        fpath = EVAL_ROOT / data_file
        if not fpath.exists():
            offending.append(f"{ds_name}: file not found: {fpath}")
            continue

        try:
            actual_hash = sha256_file(fpath)
            if actual_hash != expected_hash:
                offending.append(
                    f"{ds_name}: hash mismatch — "
                    f"expected {expected_hash}, got {actual_hash}"
                )

            actual_count = count_jsonl_items(fpath)
            if actual_count != expected_count:
                offending.append(
                    f"{ds_name}: count mismatch — "
                    f"expected {expected_count}, got {actual_count}"
                )

            pinned = PINNED_COUNTS.get(ds_name)
            if pinned is not None and actual_count != pinned:
                offending.append(
                    f"{ds_name}: pinned count mismatch — "
                    f"expected {pinned}, got {actual_count}"
                )

            if ds_name == "mbpp_sanitized" and actual_count != expected_count:
                offending.append(
                    f"{ds_name}: authoritative count mismatch — "
                    f"manifest says {expected_count}, got {actual_count}"
                )
        except Exception as exc:
            offending.append(
                f"{ds_name}: unreadable file {fpath} "
                f"({type(exc).__name__}: {exc})"
            )
            continue

    if offending:
        print("FAIL: manifest verification errors:", file=sys.stderr)
        for msg in offending:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    # Summary
    for ds_name, ds_info in datasets.items():
        print(f"  OK  {ds_name}: {ds_info['item_count']} items, hash verified")

    print(f"\nverify_manifest: ALL {len(datasets)} datasets passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
