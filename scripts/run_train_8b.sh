#!/usr/bin/env bash
# Part B pipeline: train the general (neural) expert by distilling the target
# (default Qwen3-8B), then re-run the speculative eval with the trained drafter.
#
# Stages: generate training traces -> build shards -> train general expert ->
#         eval on repetitive + code prompt sets.
#
# Tunables (env overrides):
#   REPO       target model              (default Qwen/Qwen3-8B)
#   PER_CLASS  training prompts / class  (default 60)
#   CLASSES    training classes           (default code,chat,structured,math)
#   STEPS      training steps            (default 2000)
#   SPEC_LEN   draft chain length        (default 8)
#   TARGET_ID  target variant id         (default 2 = Qwen3-8B)
#
# Usage (on the pod, after pod_setup.sh):  bash scripts/run_train_8b.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

REPO="${REPO:-Qwen/Qwen3-8B}"
PER_CLASS="${PER_CLASS:-60}"
CLASSES="${CLASSES:-code,chat,structured,math}"
STEPS="${STEPS:-2000}"
SPEC_LEN="${SPEC_LEN:-8}"
TARGET_ID="${TARGET_ID:-2}"

echo "[train_8b] repo=$REPO per_class=$PER_CLASS classes=$CLASSES steps=$STEPS"

echo "[train_8b] 1/4 generating training traces from $REPO ..."
python scripts/gen_train_data.py --repo "$REPO" --per-class "$PER_CLASS" \
  --max-new 128 --out-dir runs/traces_train --classes "$CLASSES"

echo "[train_8b] 2/4 building distillation shards ..."
python scripts/build_distill_dataset.py --traces-dir runs/traces_train \
  --out-root data/distill_train

echo "[train_8b] 3/4 training general expert (target_id=$TARGET_ID) ..."
python scripts/train_general_expert.py \
  --shards-dir data/distill_train/train --steps "$STEPS" \
  --out-dir runs/trainlogs/general_8b --target-id "$TARGET_ID"

CKPT=runs/trainlogs/general_8b/general_expert.pt
echo "[train_8b] 4/4 eval with trained drafter (spec_len=$SPEC_LEN) ..."
python scripts/run_speculative_eval.py --repo "$REPO" --general-checkpoint "$CKPT" \
  --prompt-set repetitive --spec-len "$SPEC_LEN" --max-new 128 \
  --out runs/results/integrated_8b_trained_repetitive.json
python scripts/run_speculative_eval.py --repo "$REPO" --general-checkpoint "$CKPT" \
  --prompt-set code --spec-len "$SPEC_LEN" --max-new 128 \
  --out runs/results/integrated_8b_trained_code.json

echo "[train_8b] DONE."
echo "results: runs/results/integrated_8b_trained_repetitive.json"
echo "         runs/results/integrated_8b_trained_code.json"
echo "         runs/trainlogs/general_8b/general_expert.json"