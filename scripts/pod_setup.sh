#!/usr/bin/env bash
# Idempotent RunPod pod bootstrap for familydraft.
# Safe to re-run at every pod start: it only installs what is missing and
# verifies the environment. See docs/infra.md for the pod recipe.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '[pod_setup] %s\n' "$*"; }

# --- 1. Locate a suitable Python (need >=3.11 for this project) -------------
PYTHON=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
if [ -z "$PYTHON" ]; then
  log "ERROR: no python3/python found on pod"; exit 1
fi
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "python: $PYTHON ($PYVER)"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python >=3.11 required; pod has %d.%d" % sys.version_info[:2])
PY

# --- 2. Project deps (pip-based on purpose) ---------------------------------
# RunPod PyTorch templates already ship a CUDA-matched torch. `pip install -e .`
# sees the "torch" requirement already satisfied and does NOT re-download a
# possibly-mismatched wheel. This avoids uv/torch CUDA conflicts on the pod.
cd "$REPO_DIR"
log "pip install -e . in $REPO_DIR (torch left as-is)"
"$PYTHON" -m pip install --upgrade pip --quiet || true
"$PYTHON" -m pip install -e . --no-input

# --- 2b. Optional torch upgrade (set UPGRADE_TORCH=1) -----------------------
# RunPod templates ship a working CUDA torch. Upgrade to the cu130 build only
# if you specifically want it (driver must be CUDA 13-capable, i.e. >=580).
if [ "${UPGRADE_TORCH:-0}" = "1" ]; then
  log "upgrading torch to latest +cu130 build"
  "$PYTHON" -m pip install --upgrade torch \
    --extra-index-url https://download.pytorch.org/whl/cu130
fi

# --- 3. GPU visibility -------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found - pod has no NVIDIA driver"
  exit 1
fi
GPU_LINES="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
log "nvidia-smi sees: $GPU_LINES"

if ! "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  log "ERROR: driver present but torch.cuda.is_available() is False"
  exit 1
fi

# --- 4. Version report -------------------------------------------------------
"$PYTHON" - <<'EOF'
import sys
import torch
import transformers

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"cuda runtime: {torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
print(f"gpu         : {torch.cuda.get_device_name(0)}")
import familydraft  # noqa: F401
print("familydraft : import OK")
EOF

log "pod ready"
