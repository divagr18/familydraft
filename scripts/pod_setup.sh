#!/usr/bin/env bash
# Idempotent RunPod pod bootstrap for familydraft.
# Safe to re-run at every pod start: it only installs what is missing and
# verifies the environment. See docs/infra.md for the pod recipe.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '[pod_setup] %s\n' "$*"; }

# --- 1. Basic tooling (git is preinstalled on RunPod templates) -------------
if ! command -v curl >/dev/null 2>&1; then
  log "curl missing; installing via apt (requires root)"
  apt-get update -y && apt-get install -y curl
fi

# --- 2. uv (install once, reuse on re-runs) ---------------------------------
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
else
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { log "ERROR: uv not found after install"; exit 1; }
log "uv: $(uv --version)"

# --- 3. Project deps from the committed lockfile ----------------------------
cd "$REPO_DIR"
log "uv sync --frozen in $REPO_DIR"
uv sync --frozen

# --- 4. GPU visibility -------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found - pod has no NVIDIA driver"
  exit 1
fi
GPU_LINES="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
log "nvidia-smi sees: $GPU_LINES"

if ! uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  log "ERROR: driver present but torch.cuda.is_available() is False"
  exit 1
fi

# --- 5. Version report -------------------------------------------------------
uv run python - <<'EOF'
import sys
import torch
import transformers

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"cuda runtime: {torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
print(f"gpu         : {torch.cuda.get_device_name(0)}")
EOF

log "pod ready"
