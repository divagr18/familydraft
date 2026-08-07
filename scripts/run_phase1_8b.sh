#!/usr/bin/env bash
# M3 Phase-1 pod campaign orchestrator (plan todo 22).
#
# ONE command to run the entire Phase-1 campaign on the pod:
#   setup -> 8B general-expert training (if checkpoint missing) -> full baseline
#   campaign (run_baselines.py --all) -> EAGLE-3 SpecForge setup -> aggregate
#   phase1.csv -> pre-registration order check -> verdict.
#
# Tunables (env overrides):
#   REPO          target model        (default Qwen/Qwen3-8B)
#   STEPS         general-expert training steps (default 8000)
#   RUNS          runs per row        (default 5 - matches verdict protocol)
#   MAX_NEW       tokens per prompt   (default 160)
#   MAX_PROMPTS   prompt cap per row  (default 0 = full sealed manifest)
#   SPEC_LEN      draft horizon       (default 8)
#   TARGET_ID     target variant id   (default 2 = Qwen3-8B)
#   SKIP_TRAIN    skip general-expert training if checkpoint exists (default 1)
#   EAGLE_DIR     SpecForge checkout  (default /workspace/SpecForge)
#
# Usage (on the pod, after a fresh boot):  bash scripts/run_phase1_8b.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

REPO="${REPO:-Qwen/Qwen3-8B}"
STEPS="${STEPS:-8000}"
RUNS="${RUNS:-5}"
MAX_NEW="${MAX_NEW:-160}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SPEC_LEN="${SPEC_LEN:-8}"
TARGET_ID="${TARGET_ID:-2}"
SKIP_TRAIN="${SKIP_TRAIN:-1}"
EAGLE_DIR="${EAGLE_DIR:-/workspace/SpecForge}"

log() { printf '[phase1_8b] %s\n' "$*"; }

# --- 0. Pod bootstrap (idempotent) ------------------------------------------
log "1/7 pod_setup.sh"
bash scripts/pod_setup.sh

# --- 1. 8B general-expert checkpoint (training if missing) -------------------
CKPT="runs/trainlogs/general_8b/general_expert.pt"
if [ "$SKIP_TRAIN" = "1" ] && [ -f "$CKPT" ]; then
  log "2/7 using existing general-expert checkpoint: $CKPT"
elif [ "$SKIP_TRAIN" = "1" ]; then
  log "2/7 general-expert checkpoint missing; SKIP_TRAIN=1 but nothing to use -> training anyway"
  bash scripts/run_train_8b.sh REPO="$REPO" STEPS="$STEPS" SPEC_LEN="$SPEC_LEN" TARGET_ID="$TARGET_ID"
else
  log "2/7 training general expert (steps=$STEPS)"
  bash scripts/run_train_8b.sh REPO="$REPO" STEPS="$STEPS" SPEC_LEN="$SPEC_LEN" TARGET_ID="$TARGET_ID"
fi

# --- 2. Full baseline campaign (6 systems x 4 task classes) ------------------
log "3/7 full baseline campaign (--all, runs=$RUNS, spec_len=$SPEC_LEN)"
python scripts/run_baselines.py --all \
  --repo "$REPO" \
  --runs "$RUNS" \
  --max-new "$MAX_NEW" \
  --max-prompts "$MAX_PROMPTS" \
  --spec-len "$SPEC_LEN" \
  --general-checkpoint "$CKPT" \
  --router-weights "configs/router_weights_8b.json"

# --- 3. EAGLE-3 SpecForge setup + training -----------------------------------
log "4/7 EAGLE-3 SpecForge setup"
bash scripts/setup_eagle3.sh REPO="$REPO" EAGLE_DIR="$EAGLE_DIR"
log "5/7 NOTE: run the EAGLE-3 training command printed by setup_eagle3.sh, then"
log "      produce eagle3 rows with scripts/eval_eagle3.py (see setup output)."

# --- 4. Aggregate campaign + gates -------------------------------------------
log "6/7 aggregating phase1.csv + order check"
python scripts/run_phase1.py
python scripts/m3_order_check.py
log "7/7 verdict:"
python scripts/m3_verdict.py

log "DONE. Artifacts: runs/baselines/*.json, runs/results/phase1.csv"
