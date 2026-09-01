#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

# Citrinet-144 / Citrinet-256 calibrated fair-100 suite.
#
# Every branch sees exactly 100 data epochs:
#   10 epochs from one shared CTC checkpoint + 90 branch-specific epochs.
# All branch training hyperparameters are identical:
#   seed=1, batch=64, grad accumulation=8, LR=0.025, warmup=1000, bf16.
# Only the method loss and its scale are allowed to differ.
#
# Multi-stage methods retain their stage structure within the 90 branch epochs:
#   FPKD 10+10+70, CARL 10+80.

CONFIG=student_citrinet144
EVAL_CONFIG=configs/student_citrinet144.yaml
PLAIN=data/train_clean_100.json
DENSE=data/train_clean_100.citrinet256.frame_dense_t1.json
MASS=data/train_clean_100.citrinet256.d6_mass3_ntdk_m32.json
SCTC=data/train_clean_100.citrinet256.sctc.json
CARL=data/train_clean_100.citrinet256.carl.json
CLASSIFIER=data/train_clean_100.citrinet256.carl/teacher_classifier.pt
PREFIX=arch-citrinet144-t256-lbs-cal100
WARM_NAME="${PREFIX}-shared-ctc10-s1"

COMMON=(
  --config "$CONFIG"
  --train-batch-size 64
  --accumulate-grad-batches 8
  --lr 0.025
  --warmup-steps 1000
)

require_assets() {
  local f
  for f in "$PLAIN" "$DENSE" "$MASS" "$SCTC" "$CARL" "$CLASSIFIER"; do
    require_file "$f"
  done
}

shared_warmstart() {
  require_file "$PLAIN"
  # This checkpoint is common data/model initialization, not a compared branch.
  # Its shorter warmup reaches the intended LR within the 10-epoch bootstrap.
  run_paper_train "$WARM_NAME" 10 \
    --config "$CONFIG" --manifest "$PLAIN" --kd-mode none \
    --train-batch-size 64 --accumulate-grad-batches 8 \
    --lr 0.025 --warmup-steps 500
}

warm_ckpt() {
  local ckpt
  ckpt=$(latest_last_ckpt "$WARM_NAME")
  [[ -n "$ckpt" ]] || { echo "missing shared warm-start checkpoint: $WARM_NAME" >&2; exit 1; }
  printf '%s\n' "$ckpt"
}

run_single_stage() {
  local warm
  warm=$(warm_ckpt)

  # Exact common control: same shared checkpoint and same 90-epoch continuation.
  run_paper_train "${PREFIX}-nokd-s1" 90 \
    --manifest "$PLAIN" --kd-mode none --init-ckpt "$warm" "${COMMON[@]}"

  # Calibrated from the observed Conformer:Citrinet weighted-loss ratio.
  # The method definition and targets are unchanged.
  run_paper_train "${PREFIX}-atdk-w20-ntdk6-s1" 90 \
    --manifest "$MASS" --kd-mode span_kd \
    --kd-weight 20 --span-primary-mode mass3 --span-ntdk-weight 6 \
    --init-ckpt "$warm" "${COMMON[@]}"

  # These lambda scales already matched Conformer, so retain them.
  run_paper_train "${PREFIX}-vanilla-l025-s1" 90 \
    --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode none --kd-lambda 0.25 \
    --init-ckpt "$warm" "${COMMON[@]}"

  run_paper_train "${PREFIX}-kdbe-l025-s1" 90 \
    --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode elimination --kd-lambda 0.25 \
    --init-ckpt "$warm" "${COMMON[@]}"

  run_paper_train "${PREFIX}-symmetric-l025-n4-s1" 90 \
    --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode symmetric --blank-n 4 --kd-lambda 0.25 \
    --init-ckpt "$warm" "${COMMON[@]}"

  # Guided loss was 5.4x weaker relative to CTC on Citrinet than Conformer.
  run_paper_train "${PREFIX}-guided-w5-s1" 90 \
    --manifest "$DENSE" --kd-mode guided \
    --temperature 1 --kd-weight 5 \
    --init-ckpt "$warm" "${COMMON[@]}"

  # S-CTC is frame-mean (~1) while CTC is utterance-sum (~100).  Lambda 0.98
  # keeps a nonzero CTC anchor and gives the KD term a useful relative scale.
  run_paper_train "${PREFIX}-sctc-l098-s1" 90 \
    --manifest "$SCTC" --kd-mode sctc --kd-lambda 0.98 \
    --init-ckpt "$warm" "${COMMON[@]}"

  # CR is now exposure-matched (90 continuation epochs, not 50). Its loss
  # scale and ramp are calibrated for Citrinet's shorter optimizer-step budget.
  run_paper_train "${PREFIX}-crctc-w05-r500-s1" 90 \
    --manifest "$PLAIN" --kd-mode cr_ctc \
    --cr-ctc-weight 0.5 --cr-ctc-warm-step 500 \
    --cr-ctc-time-factor 1.5 \
    --init-ckpt "$warm" "${COMMON[@]}"
}

run_multistage() {
  local warm dfkd_ckpt frkd_ckpt carl_feature_ckpt
  warm=$(warm_ckpt)

  # FPKD: same common initialization/hyperparameters, original stage order,
  # with 10+10+70 continuation epochs (100 total including shared CTC10).
  run_paper_train "${PREFIX}-fpkd-dfkd-s1" 10 \
    --manifest "$DENSE" --kd-mode fpkd_dfkd --fpkd-temp 1 \
    --init-ckpt "$warm" "${COMMON[@]}"
  dfkd_ckpt=$(latest_last_ckpt "${PREFIX}-fpkd-dfkd-s1")
  [[ -n "$dfkd_ckpt" ]] || { echo "missing FPKD DFKD checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-fpkd-frkd-s1" 10 \
    --manifest "$CARL" --kd-mode fpkd_frkd --carl-teacher-dim 640 \
    --init-ckpt "$dfkd_ckpt" "${COMMON[@]}"
  frkd_ckpt=$(latest_last_ckpt "${PREFIX}-fpkd-frkd-s1")
  [[ -n "$frkd_ckpt" ]] || { echo "missing FPKD FRKD checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-fpkd-pkd-s1" 70 \
    --manifest "$DENSE" --kd-mode fpkd_pkd \
    --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1 \
    --init-ckpt "$frkd_ckpt" "${COMMON[@]}"

  # CARL: 10+80 continuation epochs (100 total including shared CTC10).
  run_paper_train "${PREFIX}-carl-feature-s1" 10 \
    --manifest "$CARL" --kd-mode carl_feature --carl-teacher-dim 640 \
    --init-ckpt "$warm" "${COMMON[@]}"
  carl_feature_ckpt=$(latest_last_ckpt "${PREFIX}-carl-feature-s1")
  [[ -n "$carl_feature_ckpt" ]] || { echo "missing CARL feature checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-carl-full-s1" 80 \
    --manifest "$CARL" --kd-mode carl \
    --carl-classifier "$CLASSIFIER" --carl-teacher-dim 640 \
    --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
    --init-ckpt "$carl_feature_ckpt" "${COMMON[@]}"
}

eval_one() {
  local label="$1" name="$2" ckpt
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" --name "$label" --config "$EVAL_CONFIG" --beam-width 16 \
    --manifest dev_clean=data/dev_clean.json \
    --manifest dev_other=data/dev_other.json \
    --manifest test_clean=data/test_clean.json \
    --manifest test_other=data/test_other.json \
    | tee "analysis/beam16_${label}.txt"
}

evaluate_all() {
  eval_one "${PREFIX}-nokd" "${PREFIX}-nokd-s1"
  eval_one "${PREFIX}-atdk" "${PREFIX}-atdk-w20-ntdk6-s1"
  eval_one "${PREFIX}-vanilla" "${PREFIX}-vanilla-l025-s1"
  eval_one "${PREFIX}-kdbe" "${PREFIX}-kdbe-l025-s1"
  eval_one "${PREFIX}-symmetric" "${PREFIX}-symmetric-l025-n4-s1"
  eval_one "${PREFIX}-guided" "${PREFIX}-guided-w5-s1"
  eval_one "${PREFIX}-sctc" "${PREFIX}-sctc-l098-s1"
  eval_one "${PREFIX}-crctc" "${PREFIX}-crctc-w05-r500-s1"
  eval_one "${PREFIX}-fpkd" "${PREFIX}-fpkd-pkd-s1"
  eval_one "${PREFIX}-carl" "${PREFIX}-carl-full-s1"
}

case "${1:-all}" in
  warmstart) shared_warmstart ;;
  single)
    require_assets
    shared_warmstart
    run_single_stage
    ;;
  multistage)
    require_assets
    shared_warmstart
    run_multistage
    ;;
  eval) evaluate_all ;;
  all)
    require_assets
    shared_warmstart
    run_single_stage
    run_multistage
    evaluate_all
    ;;
  *)
    echo "usage: $0 [warmstart|single|multistage|eval|all]" >&2
    exit 2
    ;;
esac
