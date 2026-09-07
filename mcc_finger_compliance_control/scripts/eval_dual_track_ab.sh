#!/usr/bin/env bash
# Causal deployment validation for the current v2 obs/label-split models.
#
# Current checkpoint contract being diagnosed (not the complete dual-state
# design; see SPEED_UNIFORM_PROGRESS.md section 43):
#   observation = executed/live q + live tactile + planner
#   action      = task intent (q_ref or tangent fingertip delta)
#   execution   = action + independent FullHandMCC contact correction
#
# The old evaluation fed nominal q although the dual-track dataset was built
# from executed q_hand. It also changed the MCC preset and tactile-normal
# source in the same run. This staircase changes one factor at a time.
#
# Fast causal smoke on ep30, Variant B only:
#   EPISODES="30" MAX_STEPS=800 \
#     STAGES="S0 S1 S1_SENSOR S1_ORACLE S2_TEACHER_TACTILE S2_ORACLE S2_SENSOR" \
#     bash mcc_finger_compliance_control/scripts/eval_dual_track_ab.sh
#
# Full frozen validation after the smoke passes:
#   VARIANTS="B" EPISODES="30 45 53 59" MAX_STEPS=0 \
#     bash mcc_finger_compliance_control/scripts/eval_dual_track_ab.sh
#
# Diagnostic A/B of the two output representations:
#   VARIANTS="A B" STAGES="S1 S2_ORACLE S2_SENSOR" \
#     bash mcc_finger_compliance_control/scripts/eval_dual_track_ab.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PY=${PY:-/home/rimlab/miniconda3/envs/mjlab/bin/python}
DEPLOY=mcc_finger_compliance_control/scripts/deploy_dp_inverse.py
INVERTED=${INVERTED:-mcc_finger_compliance_control/data/inverted/mustard_v1_239_mesh_normal_inward_inverted.h5}
A_MODEL=${A_MODEL:-mcc_finger_compliance_control/data/models/mustard_v1_239_dual_track_variant_a_qref_5k}
B_MODEL=${B_MODEL:-mcc_finger_compliance_control/data/models/mustard_v1_239_dual_track_variant_b_tipdelta_5k}

EPISODES=${EPISODES:-"30 45 53 59"}
VARIANTS=${VARIANTS:-"B"}
STAGES=${STAGES:-"S0 S1 S1_SENSOR S1_ORACLE S2_TEACHER_TACTILE S2_ORACLE S2_SENSOR S3_CONTROLLER"}
MAX_STEPS=${MAX_STEPS:-0}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
DP_REPLAN_INTERVAL=${DP_REPLAN_INTERVAL:-10}

model_dir() {
  case "$1" in
    A) echo "$A_MODEL" ;;
    B) echo "$B_MODEL" ;;
    *) echo "Unknown variant '$1' (expected A or B)" >&2; return 2 ;;
  esac
}

run_stage() {
  local variant=$1
  local stage=$2
  local episode=$3
  local model
  model=$(model_dir "$variant")
  local report="$model/dual_track_${stage}_liveq_ep${episode}.csv"
  local mode=teacher_dp
  local preset=collection_matched_sensor
  local teacher_obs=teacher
  local teacher_action=dp
  local normal_source=contact_sensor
  local privileged=()

  case "$stage" in
    S0)
      # Exact recorded execution through the matched low-level stack.
      teacher_action=recorded
      ;;
    S1)
      # Teacher q/tactile conditions the DP; DP intent is executed.
      ;;
    S1_SENSOR)
      # Isolate live solver-contact point/normal while q remains teacher.
      teacher_obs=live_tactile
      ;;
    S1_ORACLE)
      # Same actual solver contact support, raw-mesh normal diagnostic.
      teacher_obs=live_tactile
      normal_source=source_mesh_oracle
      privileged=(--allow-privileged-surface-oracle)
      ;;
    S2_ORACLE)
      # Simplified live-q loop: matched MCC and privileged normal.  This
      # checkpoint has no simultaneous q_prior/compensation input stream.
      mode=live_dp
      normal_source=source_mesh_oracle
      privileged=(--allow-privileged-surface-oracle)
      ;;
    S2_TEACHER_TACTILE)
      # Pure q-feedback isolation: live q/qdot while tactile geometry and
      # tactile rates remain the time-aligned recorded teacher channels.
      # Privileged diagnostic only; not a deployable configuration.
      mode=live_dp
      teacher_obs=teacher_tactile
      ;;
    S2_SENSOR)
      # Sensor-only simplified live-q loop with collection-matched MCC.
      mode=live_dp
      ;;
    S3_CONTROLLER)
      # Only after S2 passes: transfer to the bounded 1.5 N MCC preset.
      mode=live_dp
      preset=current
      ;;
    *)
      echo "Unknown stage '$stage'" >&2
      return 2
      ;;
  esac

  echo
  echo "=== dual-track variant=$variant stage=$stage episode=$episode ==="
  echo "    q_history=live mcc=$preset tactile_normal=$normal_source"
  MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
    "$PY" "$DEPLOY" \
      --file "$INVERTED" \
      --model "$model/best.pt" \
      --episode-id "$episode" \
      --mode "$mode" \
      --execution-layer fullhand_mcc \
      --mcc-preset "$preset" \
      --dp-history-q-source live \
      --teacher-observation-source "$teacher_obs" \
      --teacher-action-source "$teacher_action" \
      --dp-tactile-normal-source "$normal_source" \
      --chunk-execution \
      --dp-replan-interval "$DP_REPLAN_INTERVAL" \
      --viewer headless \
      --device "$DEVICE" \
      --seed "$SEED" \
      --max-steps "$MAX_STEPS" \
      --report "$report" \
      "${privileged[@]}"
}

for variant in $VARIANTS; do
  for stage in $STAGES; do
    for episode in $EPISODES; do
      run_stage "$variant" "$stage" "$episode"
    done
  done
done

echo
echo "[DONE] Dual-track causal staircase completed."
