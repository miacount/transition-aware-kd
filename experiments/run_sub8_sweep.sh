#!/usr/bin/env bash
# Run full 8x subsampling KD sweep sequentially.
# no-KD already done; this runs trans (x3) then logit (x3).
# Usage: bash experiments/run_sub8_sweep.sh
set -euo pipefail

TRANS_MANIFEST=data/train_clean_100.json
LOGIT_MANIFEST=data/train_clean_100.small_teacher.frame_top8_t1.json

[[ -f "$TRANS_MANIFEST" ]] || { echo "missing $TRANS_MANIFEST" >&2; exit 1; }
[[ -f "$LOGIT_MANIFEST" ]] || {
  echo "missing $LOGIT_MANIFEST — run experiments/presets/20_build_frame_topk_targets.sh first" >&2
  exit 1
}

run() {
  echo ""
  echo "════════════════════════════════════════"
  echo "  $*"
  echo "════════════════════════════════════════"
  bash experiments/train.sh "$@"
}

# ── trans KD (best weight from 4x) ───────────────────────────────────────────
run --config student_sub8 \
    --name "student-sub8-trans-w0.25-clean" \
    --manifest "$TRANS_MANIFEST" \
    --kd-mode trans \
    --kd-weight 0.25

# ── logit KD (best weight from 4x) ───────────────────────────────────────────
run --config student_sub8 \
    --name "student-sub8-logit-top8-t1-w10.0-clean" \
    --manifest "$LOGIT_MANIFEST" \
    --kd-mode logit \
    --kd-weight 10.0

echo ""
echo "All sub8 sweep runs complete."
