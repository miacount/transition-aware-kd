#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

# Training-recipe-only Citrinet study.
# Keep the proposed method fixed:
#   teacher=Citrinet-256, primary=Mass3, kd_weight=12.5,
#   conditional NTDK weight=4, and the existing cached targets.
# Only the optimizer warmup and the time at which KD is enabled vary.

PLAIN_MANIFEST=data/train_clean_100.json
KD_MANIFEST=data/train_clean_100.citrinet256.d6_mass3_ntdk_m32.json
CONFIG=student_citrinet144
EVAL_CONFIG=configs/student_citrinet144.yaml

COMMON_TRAIN_ARGS=(
  --config "$CONFIG"
  --epochs 100
  --train-batch-size 64
  --accumulate-grad-batches 8
  --lr 0.025
  --warmup-steps 500
)

eval_run() {
  local name="$1"
  local label="$2"
  local ckpt
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  python scripts/evaluate_student.py \
    --config "$EVAL_CONFIG" \
    --ckpt "$ckpt" \
    --manifest dev_clean=data/dev_clean.json \
    --manifest dev_other=data/dev_other.json \
    --manifest test_clean=data/test_clean.json \
    --manifest test_other=data/test_other.json \
    | tee "analysis/eval_${label}.txt"
}

run_nokd() {
  local name=arch-citrinet144-t256-lbs-nokd-b64a8-lr025-wu500-e100-s1
  run_paper_train "$name" 100 \
    --manifest "$PLAIN_MANIFEST" \
    --kd-mode none \
    "${COMMON_TRAIN_ARGS[@]}" \
    2>&1 | tee analysis/citrinet144_recipe_nokd_wu500.log
  eval_run "$name" "$name"
}

run_atdk_wu500() {
  local name=arch-citrinet144-t256-lbs-atdk-w12p5-ntdk4-b64a8-lr025-wu500-e100-s1
  run_paper_train "$name" 100 \
    --manifest "$KD_MANIFEST" \
    --kd-mode span_kd \
    --kd-weight 12.5 \
    --span-primary-mode mass3 \
    --span-ntdk-weight 4 \
    "${COMMON_TRAIN_ARGS[@]}" \
    2>&1 | tee analysis/citrinet144_recipe_atdk_wu500.log
  eval_run "$name" "$name"
}

run_atdk_delayed() {
  local name=arch-citrinet144-t256-lbs-atdk-w12p5-ntdk4-b64a8-lr025-wu500-kdstart500-e100-s1
  run_paper_train "$name" 100 \
    --manifest "$KD_MANIFEST" \
    --kd-mode span_kd \
    --kd-weight 12.5 \
    --span-primary-mode mass3 \
    --span-ntdk-weight 4 \
    --kd-start-step 500 \
    "${COMMON_TRAIN_ARGS[@]}" \
    2>&1 | tee analysis/citrinet144_recipe_atdk_wu500_kdstart500.log
  eval_run "$name" "$name"
}

case "${1:-all}" in
  nokd) run_nokd ;;
  atdk) run_atdk_wu500 ;;
  delayed) run_atdk_delayed ;;
  all)
    run_nokd
    run_atdk_wu500
    run_atdk_delayed
    ;;
  *)
    echo "usage: $0 [nokd|atdk|delayed|all]" >&2
    exit 2
    ;;
esac
