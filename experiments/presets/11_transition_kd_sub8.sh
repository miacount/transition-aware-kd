#!/usr/bin/env bash
# Transition KD sweep with 8x subsampling student.
# Usage: bash experiments/presets/11_transition_kd_sub8.sh [kd_weight]
# default weight: 0.25  (same as best 4x result for direct comparison)
set -euo pipefail

MANIFEST=${MANIFEST:-data/train_clean_100.json}
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

KD_WEIGHT=${1:-0.25}

bash experiments/train.sh \
  --config student_sub8 \
  --name "student-sub8-trans-w${KD_WEIGHT}-clean" \
  --manifest "$MANIFEST" \
  --kd-mode trans \
  --kd-weight "$KD_WEIGHT"
