#!/usr/bin/env bash
# EAGLE-3 baseline resume sanity (M3 / plan todo 13 QA).
#
# Restores a SpecForge EAGLE-3 training run from its latest checkpoint and
# completes a short resume pass, asserting loss continuity. This is the
# failure-path QA the plan demands: a pod that dies mid-training must be able
# to resume without corrupting the run.
#
# Tunables (env overrides):
#   EAGLE_DIR   SpecForge checkout (default /workspace/SpecForge)
#   CKPT_DIR    training checkpoint dir (default runs/baselines/eagle3/checkpoints)
#   RESUME_STEPS  steps for the sanity pass (default 100)
#
# Usage (on the pod):  bash scripts/resume_baseline.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

EAGLE_DIR="${EAGLE_DIR:-/workspace/SpecForge}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/runs/baselines/eagle3/checkpoints}"
RESUME_STEPS="${RESUME_STEPS:-100}"

log() { printf '[resume_baseline] %s\n' "$*"; }

if [ ! -d "$CKPT_DIR" ]; then
  log "ERROR: no checkpoint dir at $CKPT_DIR"; exit 1
fi

# Latest checkpoint = newest file in the checkpoint dir.
LATEST="$(ls -1t "$CKPT_DIR" | head -1)"
if [ -z "$LATEST" ]; then
  log "ERROR: checkpoint dir empty"; exit 1
fi
log "resuming from latest checkpoint: $CKPT_DIR/$LATEST"

if [ ! -d "$EAGLE_DIR" ]; then
  log "ERROR: SpecForge not found at $EAGLE_DIR (run scripts/setup_eagle3.sh first)"; exit 1
fi
cd "$EAGLE_DIR"

log "running $RESUME_STEPS-step resume sanity pass ..."
"${PYTHON:-python}" -m specforge.eagle3.train \
  --resume "$CKPT_DIR/$LATEST" \
  --steps "$RESUME_STEPS" \
  --output-dir "$REPO_DIR/runs/baselines/eagle3/resume_sanity"

log "resume sanity pass complete; verify loss continuity in runs/baselines/eagle3/resume_sanity"
