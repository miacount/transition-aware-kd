#!/usr/bin/env bash
# Logit KD sweep with 8x subsampling student.
# Usage: bash experiments/presets/21_logit_kd_sub8.sh [kd_weight]
# default weight: 10.0  (same as best 4x result for direct comparison)
# NOTE: frame targets built at teacher 4x rate must be resampled to student 8x
#       length at training time — intentional misalignment being studied.
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.frame_top8_t1.json
[[ -f "$MANIFEST" ]] || {
  echo "missing $MANIFEST; run experiments/presets/20_build_frame_topk_targets.sh first" >&2
  exit 1
}

KD_WEIGHT=${1:-10.0}

bash experiments/train.sh \
  --config student_sub8 \
  --name "student-sub8-logit-top8-t1-w${KD_WEIGHT}-clean" \
  --manifest "$MANIFEST" \
  --kd-mode logit \
  --kd-weight "$KD_WEIGHT"
