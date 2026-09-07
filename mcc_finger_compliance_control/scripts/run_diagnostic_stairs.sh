#!/usr/bin/env bash
# Compatibility entry point. The old staircase used nominal-q history for a
# model trained on executed q_hand and therefore did not test the dual-track
# contract. Keep one canonical protocol to prevent the two scripts diverging.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$ROOT/mcc_finger_compliance_control/scripts/eval_dual_track_ab.sh" "$@"
