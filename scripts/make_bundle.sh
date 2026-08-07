#!/usr/bin/env bash
# Package a portable FamilyDraftMoE bundle (tar.gz) excluding heavy/local dirs
# (.git, .venv, data, runs, .codegraph). Upload the bundle to the RunPod pod
# (network volume) instead of pushing to git, if you prefer a no-git path.
#
# Usage:  bash scripts/make_bundle.sh [output_path]
# Output: familydraft_bundle.tar.gz (default in repo parent dir)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
OUT="${1:-$(dirname "$REPO_DIR")/familydraft_bundle.tar.gz}"

cd "$REPO_DIR"
tar --exclude='.git' \
    --exclude='.venv' \
    --exclude='data' \
    --exclude='runs' \
    --exclude='.codegraph' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='*.egg-info' \
    -czf "$OUT" -C "$(dirname "$REPO_DIR")" "$REPO_NAME"

echo "bundle written: $OUT"
echo "size: $(du -h "$OUT" | cut -f1)"
echo
echo "On the pod, upload it (e.g. to /workspace) then:"
echo "  cd /workspace && tar xzf $(basename "$OUT") && cd $REPO_NAME"
echo "  bash scripts/pod_setup.sh && bash scripts/run_8b.sh"
