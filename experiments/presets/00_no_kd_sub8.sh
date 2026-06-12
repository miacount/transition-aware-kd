#!/usr/bin/env bash
# No-KD baseline with 8x subsampling student (CTC only).
set -euo pipefail

MANIFEST=${1:-data/train_clean_100.json}
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

bash experiments/train.sh \
  --config student_sub8 \
  --name "student-sub8-no-kd-clean" \
  --manifest "$MANIFEST" \
  --kd-mode none \
  --kd-weight 0.0
