#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

# ATDK-only lower-scale verification for Citrinet-144 / Citrinet-256.
# Every candidate starts from the same frozen CTC10 checkpoint and receives
# exactly 120 continuation epochs with the same optimizer schedule. Selection
# is dev-clean only; held-out/test manifests are deliberately disabled.

CONFIG=student_citrinet144
MASS=data/train_clean_100.citrinet256.d6_mass3_ntdk_m32.json
WARM_NAME=arch-citrinet144-t256-lbs-cal100-shared-ctc10-s1
PREFIX=arch-citrinet144-t256-lbs-atdk-scale120

COMMON=(
  --config "$CONFIG"
  --manifest "$MASS"
  --kd-mode span_kd
  --span-primary-mode mass3
  --train-batch-size 64
  --accumulate-grad-batches 8
  --lr 0.025
  --warmup-steps 1000
  --dev-only
  --save-top-k 1
)

require_file "$MASS"
warm=$(latest_last_ckpt "$WARM_NAME")
[[ -n "$warm" ]] || {
  echo "missing shared warm-start checkpoint: $WARM_NAME" >&2
  exit 1
}

run_candidate() {
  local tag="$1" primary="$2" dark="$3"
  run_paper_train "${PREFIX}-${tag}-s1" 120 \
    --kd-weight "$primary" --span-ntdk-weight "$dark" \
    --init-ckpt "$warm" "${COMMON[@]}"
}

run_candidate p2p5-d0p75 2.5 0.75
run_candidate p5-d1p5 5 1.5

echo "[$(date -u +'%F %T UTC')] ATDK scale120 dev-only sweep complete"
