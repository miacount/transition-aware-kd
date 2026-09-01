#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

# Frozen seed-1 fair-comparison run after equal-budget dev-only tuning.
# All branches share CTC10. Standard branches use 120 continuation epochs.
# CR-CTC uses 60 epochs (two encoder forwards/step) for forward-compute matching.
# S-CTC keeps its required KD -> supervised CTC recipe (96+24).
# FPKD/CARL keep their method stages but use stage-length-proportional warmups.
# No held-out/test evaluation occurs until every recipe and scale is frozen.

CONFIG=student_citrinet144
EVAL_CONFIG=configs/student_citrinet144.yaml
PLAIN=data/train_clean_100.json
DENSE=data/train_clean_100.citrinet256.frame_dense_t1.json
MASS=data/train_clean_100.citrinet256.d6_mass3_ntdk_m32.json
SCTC=data/train_clean_100.citrinet256.sctc.json
CARL=data/train_clean_100.citrinet256.carl.json
CLASSIFIER=data/train_clean_100.citrinet256.carl/teacher_classifier.pt
WARM_NAME=arch-citrinet144-t256-lbs-cal100-shared-ctc10-s1
PREFIX=arch-citrinet144-t256-lbs-fairv3
SELECTION=analysis/citrinet144_tune30_selection.tsv

COMMON=(
  --config "$CONFIG"
  --train-batch-size 64
  --accumulate-grad-batches 8
  --lr 0.025
  --dev-only
)

require_assets() {
  local file
  for file in "$PLAIN" "$DENSE" "$MASS" "$SCTC" "$CARL" "$CLASSIFIER" "$SELECTION"; do
    require_file "$file"
  done
}

warm_ckpt() {
  local ckpt
  ckpt=$(latest_last_ckpt "$WARM_NAME")
  [[ -n "$ckpt" ]] || { echo "missing shared warm-start checkpoint: $WARM_NAME" >&2; exit 1; }
  printf '%s\n' "$ckpt"
}

selected_scale() {
  local method="$1" value
  value=$(awk -F '\t' -v m="$method" '$1==m {print $3; exit}' "$SELECTION")
  [[ -n "$value" ]] || { echo "missing selected scale for $method in $SELECTION" >&2; exit 1; }
  printf '%s\n' "$value"
}

run_single_stage() {
  local warm scale primary dark
  warm=$(warm_ckpt)

  run_paper_train "${PREFIX}-nokd-s1" 120 --manifest "$PLAIN" --kd-mode none \
    --init-ckpt "$warm" --warmup-steps 1000 "${COMMON[@]}"

  scale=$(selected_scale atdk)
  read -r primary dark <<<"$scale"
  run_paper_train "${PREFIX}-atdk-s1" 120 --manifest "$MASS" --kd-mode span_kd \
    --kd-weight "$primary" --span-primary-mode mass3 --span-ntdk-weight "$dark" \
    --init-ckpt "$warm" --warmup-steps 1000 "${COMMON[@]}"

  scale=$(selected_scale vanilla)
  run_paper_train "${PREFIX}-vanilla-s1" 120 --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum --blank-mode none \
    --kd-lambda "$scale" --init-ckpt "$warm" --warmup-steps 1000 "${COMMON[@]}"

  scale=$(selected_scale kdbe)
  run_paper_train "${PREFIX}-kdbe-s1" 120 --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum --blank-mode elimination \
    --kd-lambda "$scale" --init-ckpt "$warm" --warmup-steps 1000 "${COMMON[@]}"

  scale=$(selected_scale symmetric)
  run_paper_train "${PREFIX}-symmetric-s1" 120 --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric --blank-n 4 \
    --kd-lambda "$scale" --init-ckpt "$warm" --warmup-steps 1000 "${COMMON[@]}"

  scale=$(selected_scale guided)
  run_paper_train "${PREFIX}-guided-s1" 120 --manifest "$DENSE" --kd-mode guided \
    --temperature 1 --kd-weight "$scale" \
    --init-ckpt "$warm" --warmup-steps 1000 "${COMMON[@]}"
}

run_sctc() {
  local warm scale kd_name kd_ckpt
  warm=$(warm_ckpt)
  scale=$(selected_scale sctc)
  kd_name="${PREFIX}-sctc-kd-s1"
  run_paper_train "$kd_name" 96 --manifest "$SCTC" --kd-mode sctc \
    --kd-lambda "$scale" --init-ckpt "$warm" --warmup-steps 960 "${COMMON[@]}"
  kd_ckpt=$(latest_last_ckpt "$kd_name")
  [[ -n "$kd_ckpt" ]] || { echo "missing final S-CTC KD checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-sctc-ctcft-s1" 24 --manifest "$PLAIN" --kd-mode none \
    --init-ckpt "$kd_ckpt" --warmup-steps 240 "${COMMON[@]}"
}

run_crctc() {
  local warm scale
  warm=$(warm_ckpt)
  scale=$(selected_scale crctc)
  run_paper_train "${PREFIX}-crctc-compute60-s1" 60 --manifest "$PLAIN" --kd-mode cr_ctc \
    --cr-ctc-weight "$scale" --cr-ctc-warm-step 333 --cr-ctc-time-factor 1.5 \
    --init-ckpt "$warm" --warmup-steps 500 "${COMMON[@]}"
}

run_fpkd() {
  local warm scale dfkd_name dfkd_ckpt frkd_name frkd_ckpt
  warm=$(warm_ckpt)
  scale=$(selected_scale fpkd)
  dfkd_name="${PREFIX}-fpkd-dfkd-s1"
  frkd_name="${PREFIX}-fpkd-frkd-s1"
  run_paper_train "$dfkd_name" 10 --manifest "$DENSE" --kd-mode fpkd_dfkd \
    --fpkd-temp 1 --init-ckpt "$warm" --warmup-steps 100 "${COMMON[@]}"
  dfkd_ckpt=$(latest_last_ckpt "$dfkd_name")
  [[ -n "$dfkd_ckpt" ]] || { echo "missing final FPKD DFKD checkpoint" >&2; exit 1; }
  run_paper_train "$frkd_name" 10 --manifest "$CARL" --kd-mode fpkd_frkd \
    --carl-teacher-dim 640 --init-ckpt "$dfkd_ckpt" --warmup-steps 100 "${COMMON[@]}"
  frkd_ckpt=$(latest_last_ckpt "$frkd_name")
  [[ -n "$frkd_ckpt" ]] || { echo "missing final FPKD FRKD checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-fpkd-pkd-s1" 100 --manifest "$DENSE" --kd-mode fpkd_pkd \
    --fpkd-bkl-weight "$scale" --fpkd-nbf-weight "$scale" --fpkd-temp 1 \
    --init-ckpt "$frkd_ckpt" --warmup-steps 1000 "${COMMON[@]}"
}

run_carl() {
  local warm scale feature_name feature_ckpt
  warm=$(warm_ckpt)
  scale=$(selected_scale carl)
  feature_name="${PREFIX}-carl-feature-s1"
  run_paper_train "$feature_name" 10 --manifest "$CARL" --kd-mode carl_feature \
    --carl-teacher-dim 640 --init-ckpt "$warm" --warmup-steps 100 "${COMMON[@]}"
  feature_ckpt=$(latest_last_ckpt "$feature_name")
  [[ -n "$feature_ckpt" ]] || { echo "missing final CARL feature checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-carl-full-s1" 110 --manifest "$CARL" --kd-mode carl \
    --carl-classifier "$CLASSIFIER" --carl-teacher-dim 640 \
    --carl-alpha "$scale" --carl-gamma "$scale" --carl-lambda "$scale" \
    --init-ckpt "$feature_ckpt" --warmup-steps 1100 "${COMMON[@]}"
}

eval_one() {
  local label="$1" name="$2" ckpt
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing evaluation checkpoint: $name" >&2; exit 1; }
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
  eval_one "${PREFIX}-atdk" "${PREFIX}-atdk-s1"
  eval_one "${PREFIX}-vanilla" "${PREFIX}-vanilla-s1"
  eval_one "${PREFIX}-kdbe" "${PREFIX}-kdbe-s1"
  eval_one "${PREFIX}-symmetric" "${PREFIX}-symmetric-s1"
  eval_one "${PREFIX}-guided" "${PREFIX}-guided-s1"
  eval_one "${PREFIX}-sctc" "${PREFIX}-sctc-ctcft-s1"
  eval_one "${PREFIX}-crctc" "${PREFIX}-crctc-compute60-s1"
  eval_one "${PREFIX}-fpkd" "${PREFIX}-fpkd-pkd-s1"
  eval_one "${PREFIX}-carl" "${PREFIX}-carl-full-s1"
}

require_assets
run_single_stage
run_sctc
run_crctc
run_fpkd
run_carl
evaluate_all
