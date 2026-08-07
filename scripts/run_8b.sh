#!/usr/bin/env bash
# Run the FamilyDraftMoE integrated speculative speedup eval on Qwen3-8B.
#
# Assumes pod_setup.sh has already run (deps installed, GPU verified).
# Writes results to runs/results/integrated_speedup_8b.json and prints a summary.
#
# Tunables (env overrides):
#   REPO       target model           (default Qwen/Qwen3-8B)
#   MAX_NEW    tokens per prompt      (default 128)
#   SPEC_LEN   draft chain length     (default 8)
#   DRAFTERS   comma list             (default copy,general ; use "copy" to skip general)
#
# Usage (on the pod):  bash scripts/run_8b.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REPO="${REPO:-Qwen/Qwen3-8B}"
MAX_NEW="${MAX_NEW:-128}"
SPEC_LEN="${SPEC_LEN:-8}"
DRAFTERS="${DRAFTERS:-copy,general}"

echo "[run_8b] target=$REPO max_new=$MAX_NEW spec_len=$SPEC_LEN drafters=$DRAFTERS"
python - <<'PY'
import torch
print("device:", torch.cuda.get_device_name(0),
      "| vram(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

python scripts/run_speculative_eval.py \
  --repo "$REPO" \
  --max-new "$MAX_NEW" \
  --spec-len "$SPEC_LEN" \
  --drafters "$DRAFTERS" \
  --out runs/results/integrated_speedup_8b.json

echo "[run_8b] DONE. Results: runs/results/integrated_speedup_8b.json"
