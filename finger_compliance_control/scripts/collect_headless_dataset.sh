#!/usr/bin/env bash
set -euo pipefail

# Bulk headless data collection.
# Default mode: single-file output, target ~= 12,800,000 transitions.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-/home/rimlab/miniconda3/envs/mjlab/bin/python}
TASK_ID=${TASK_ID:-Leaphand-Finger-Compliance-Control}
OUT_DIR=${OUT_DIR:-finger_compliance_control/data/headless}
RUN_TAG=${RUN_TAG:-headless_train_$(date +%Y%m%d_%H%M%S)}

NUM_ENVS=${NUM_ENVS:-16}
STEPS_PER_SHARD=${STEPS_PER_SHARD:-5000}
TARGET_TRANSITIONS=${TARGET_TRANSITIONS:-12800000}
ACTION_NOISE_STD=${ACTION_NOISE_STD:-0.02}
CCD_ITERATIONS=${CCD_ITERATIONS:-1000}
FSR_SOURCE=${FSR_SOURCE:-sensor}
SINGLE_FILE=${SINGLE_FILE:-1}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

transitions_per_shard=$((NUM_ENVS * STEPS_PER_SHARD))
num_shards=$(((TARGET_TRANSITIONS + transitions_per_shard - 1) / transitions_per_shard))
total_steps_single=$(((TARGET_TRANSITIONS + NUM_ENVS - 1) / NUM_ENVS))

printf "[INFO] task=%s\n" "$TASK_ID"
printf "[INFO] run_tag=%s\n" "$RUN_TAG"
printf "[INFO] out_dir=%s\n" "$OUT_DIR"
printf "[INFO] num_envs=%d\n" "$NUM_ENVS"
printf "[INFO] steps_per_shard=%d\n" "$STEPS_PER_SHARD"
printf "[INFO] transitions_per_shard=%d\n" "$transitions_per_shard"
printf "[INFO] target_transitions=%d\n" "$TARGET_TRANSITIONS"
printf "[INFO] single_file=%s\n" "$SINGLE_FILE"

if [[ "$SINGLE_FILE" == "1" ]]; then
  echo "[COLLECT] single file -> ${OUT_DIR}/${RUN_TAG}.h5"
  echo "[INFO] total_steps=${total_steps_single} (num_envs=${NUM_ENVS})"

  PYTHONPATH=src "$PYTHON_BIN" finger_compliance_control/scripts/collect_data_headless.py "$TASK_ID" \
    --viewer headless \
    --output-dir "$OUT_DIR" \
    --filename "$RUN_TAG" \
    --num-envs "$NUM_ENVS" \
    --total-steps "$total_steps_single" \
    --action-noise-std "$ACTION_NOISE_STD" \
    --ccd-iterations "$CCD_ITERATIONS" \
    --fsr-source "$FSR_SOURCE" \
    --randomize-object-profile True \
    --randomize-object-orientation True

  echo "[DONE] Collected about $((NUM_ENVS * total_steps_single)) transitions in one file"
else
  printf "[INFO] steps_per_shard=%d\n" "$STEPS_PER_SHARD"
  printf "[INFO] transitions_per_shard=%d\n" "$transitions_per_shard"
  printf "[INFO] shards=%d\n" "$num_shards"

  for ((i=1; i<=num_shards; i++)); do
    shard_name=$(printf "%s_s%03d" "$RUN_TAG" "$i")
    echo "[COLLECT] shard $i/$num_shards -> ${OUT_DIR}/${shard_name}.h5"

    PYTHONPATH=src "$PYTHON_BIN" finger_compliance_control/scripts/collect_data_headless.py "$TASK_ID" \
      --viewer headless \
      --output-dir "$OUT_DIR" \
      --filename "$shard_name" \
      --num-envs "$NUM_ENVS" \
      --total-steps "$STEPS_PER_SHARD" \
      --action-noise-std "$ACTION_NOISE_STD" \
      --ccd-iterations "$CCD_ITERATIONS" \
      --fsr-source "$FSR_SOURCE" \
      --randomize-object-profile True \
      --randomize-object-orientation True
  done

  echo "[DONE] Collected about $((num_shards * transitions_per_shard)) transitions in $num_shards shards"
fi
