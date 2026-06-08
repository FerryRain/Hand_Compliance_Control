#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

PYTHON_BIN=/home/rimlab/miniconda3/envs/mjlab/bin/python
SCRIPT=finger_compliance_control/scripts/train_goal_conditioned.py
H5=finger_compliance_control/data/headless/collect_20260407_210549.h5
SAVE_DIR=finger_compliance_control/data/models
LOG_DIR=finger_compliance_control/data/models/logs
LAUNCHER_LOG_DIR=finger_compliance_control/data/models/launch_logs

EPOCHS=20
BATCH_SIZE=4096
LR=1e-4
WEIGHT_DECAY=1e-4
SEED=42
NUM_WORKERS=8
RUN_TAG=final_v1_resume
RESUME_CHECKPOINT=finger_compliance_control/data/models/final_v1_e40_residual_qw0.1_qstep5_s42.pt

QUALITY_LOSS_WEIGHTS=(0.1)
QUALITY_TARGET_STEPS=(5)

for quality_loss_weight in "${QUALITY_LOSS_WEIGHTS[@]}"; do
  for quality_target_step in "${QUALITY_TARGET_STEPS[@]}"; do
    run_name="${RUN_TAG}_e${EPOCHS}_residual_qw${quality_loss_weight}_qstep${quality_target_step}_s${SEED}"
    log_name="${run_name}.log"
    launcher_log="${LAUNCHER_LOG_DIR}/${run_name}.txt"

    cd "$REPO_ROOT"
    mkdir -p "$LAUNCHER_LOG_DIR"
    {
      echo "timestamp=$(date -Iseconds)"
      echo "run_name=$run_name"
      echo "h5=$H5"
      echo "batch_size=$BATCH_SIZE"
      echo "num_workers=$NUM_WORKERS"
      echo "quality_loss_weight=$quality_loss_weight"
      echo "quality_target_step=$quality_target_step"
      echo "command=PYTHONPATH=src $PYTHON_BIN $SCRIPT --h5 $H5 --device cuda --drop-palm-fsr --window 20 --epochs $EPOCHS --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --lr $LR --weight-decay $WEIGHT_DECAY --quality-loss-weight $quality_loss_weight --quality-target-step $quality_target_step --predict-residual --seed $SEED --name $run_name --save-dir $SAVE_DIR --log-dir $LOG_DIR --log-name $log_name"
    } > "$launcher_log"

    echo "[launch] $run_name"
    echo "[launch log] $launcher_log"
    PYTHONPATH=src "$PYTHON_BIN" "$SCRIPT" \
      --h5 "$H5" \
      --device cuda \
      --drop-palm-fsr \
      --window 20 \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --quality-loss-weight "$quality_loss_weight" \
      --quality-target-step "$quality_target_step" \
      --predict-residual \
      --resume-checkpoint "$RESUME_CHECKPOINT" \
      --resume-optimizer \
      --seed "$SEED" \
      --name "$run_name" \
      --save-dir "$SAVE_DIR" \
      --log-dir "$LOG_DIR" \
      --log-name "$log_name"
  done
done