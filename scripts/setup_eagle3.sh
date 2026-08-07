#!/usr/bin/env bash
# EAGLE-3 baseline setup via SpecForge (M3 / plan todo 13).
#
# Clones sgl-project/SpecForge @ 7d5a693 (MIT), installs its deps, and prints
# the EAGLE-3 training command for the pod target. The actual training runs on
# the RunPod A100 pod (Qwen3-8B); this script is idempotent and safe to re-run.
#
# Tunables (env overrides):
#   REPO       target model to train the EAGLE-3 drafter against (default Qwen/Qwen3-8B)
#   SPECFORGE_SHA  pinned commit (default 7d5a693)
#   EAGLE_DIR  SpecForge checkout location (default /workspace/SpecForge)
#
# Usage (on the pod, after pod_setup.sh):  bash scripts/setup_eagle3.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

REPO="${REPO:-Qwen/Qwen3-8B}"
SPECFORGE_SHA="${SPECFORGE_SHA:-7d5a693}"
EAGLE_DIR="${EAGLE_DIR:-/workspace/SpecForge}"

log() { printf '[setup_eagle3] %s\n' "$*"; }

if ! command -v git >/dev/null 2>&1; then
  log "ERROR: git not found on pod"; exit 1
fi

# --- 1. Clone / update SpecForge at the pinned commit ----------------------
if [ ! -d "$EAGLE_DIR" ]; then
  log "cloning SpecForge into $EAGLE_DIR ..."
  git clone https://github.com/sgl-project/SpecForge.git "$EAGLE_DIR"
fi
cd "$EAGLE_DIR"
git fetch --tags origin || true
git checkout "$SPECFORGE_SHA"
log "SpecForge pinned at $(git rev-parse --short HEAD)"

# --- 2. Install SpecForge deps (torch left as-is; pod already has CUDA torch) --
"${PYTHON:-python}" -m pip install -e . --no-input || {
  log "editable install failed; trying requirements-only"
  "${PYTHON:-python}" -m pip install -r requirements.txt --no-input || true
}

# --- 3. Verify the EAGLE-3 entry point exists --------------------------------
if ! "${PYTHON:-python}" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('eagle') else 1)"; then
  log "WARNING: no top-level 'eagle' package importable; will rely on SpecForge CLI"
else
  log "eagle package importable"
fi

# --- 4. Print the training command (not run here; pod executes it) -----------
cat <<EOF

[setup_eagle3] EAGLE-3 training command for REPO=$REPO:
  cd $EAGLE_DIR && python -m specforge.eagle3.train \\
    --target $REPO \\
    --output-dir $REPO_DIR/runs/baselines/eagle3_$REPO \\
    --checkpoint-dir $REPO_DIR/runs/baselines/eagle3_$REPO/checkpoints

On completion, produce the baseline report with:
  python scripts/eval_eagle3.py --repo $REPO \\
    --checkpoint $REPO_DIR/runs/baselines/eagle3_$REPO/checkpoints \\
    --out runs/baselines/eagle3_specforge_<task>.json
EOF

log "setup complete"
