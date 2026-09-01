#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

# Citrinet-144 / Citrinet-256 fair 100-epoch suite on LibriSpeech train-clean-100.
# Shared environment:
#   seed=1, physical batch=64, grad accumulation=8 (effective batch=512),
#   peak LR=0.025, warmup=1000 optimizer steps, bf16, identical tokenizer/data.
# Method-specific multi-stage protocols retain their original total exposure:
#   S-CTC 80+20, FPKD 10+10+80, CARL 10+90.
# CR-CTC uses 50 epochs because it processes two augmented views per batch.

CONFIG=student_citrinet144
EVAL_CONFIG=configs/student_citrinet144.yaml
PLAIN=data/train_clean_100.json
DENSE=data/train_clean_100.citrinet256.frame_dense_t1.json
MASS=data/train_clean_100.citrinet256.d6_mass3_ntdk_m32.json
SCTC=data/train_clean_100.citrinet256.sctc.json
CARL=data/train_clean_100.citrinet256.carl.json
CLASSIFIER=data/train_clean_100.citrinet256.carl/teacher_classifier.pt
PREFIX=arch-citrinet144-t256-lbs-fair100

COMMON=(
  --config "$CONFIG"
  --train-batch-size 64
  --accumulate-grad-batches 8
  --lr 0.025
  --warmup-steps 1000
)

eval_one() {
  local label="$1"
  local name="$2"
  local ckpt
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" \
    --name "$label" \
    --config "$EVAL_CONFIG" \
    --beam-width 16 \
    --manifest dev_clean=data/dev_clean.json \
    --manifest dev_other=data/dev_other.json \
    --manifest test_clean=data/test_clean.json \
    --manifest test_other=data/test_other.json \
    | tee "analysis/beam16_${label}.txt"
}

run_ours() {
  # Proposed method is unchanged: same cached Mass3/NTDK targets and weights.
  # Only the validated CTC-only initialization window is applied.
  require_file "$MASS"
  run_paper_train "${PREFIX}-atdk-w12p5-ntdk4-wu1000-kdstart500-s1" 100 \
    --manifest "$MASS" \
    --kd-mode span_kd \
    --kd-weight 12.5 \
    --span-primary-mode mass3 \
    --span-ntdk-weight 4 \
    --kd-start-step 500 \
    "${COMMON[@]}"
}

prepare_full_targets() {
  if [[ -f "$SCTC" && -f "$CARL" && -f "$CLASSIFIER" ]]; then
    echo "=== [reuse] Citrinet-256 S-CTC/CARL targets are complete ==="
    return 0
  fi
  TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-16}" \
    bash experiments/prepare_citrinet256_teacher_targets.sh lbs full
  require_file "$SCTC"
  require_file "$CARL"
  require_file "$CLASSIFIER"
}

run_baselines() {
  for f in "$PLAIN" "$DENSE" "$SCTC" "$CARL" "$CLASSIFIER"; do
    require_file "$f"
  done

  # Strong existing no-KD control with the exact shared recipe.
  run_paper_train arch-citrinet144-t256-lbs-nokd-b64a8-lr025-e100-s1 100 \
    --manifest "$PLAIN" --kd-mode none "${COMMON[@]}"

  run_paper_train "${PREFIX}-vanilla-l025-s1" 100 \
    --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode none --kd-lambda 0.25 "${COMMON[@]}"

  run_paper_train "${PREFIX}-kdbe-l025-s1" 100 \
    --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode elimination --kd-lambda 0.25 "${COMMON[@]}"

  run_paper_train "${PREFIX}-symmetric-l025-n4-s1" 100 \
    --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode symmetric --blank-n 4 --kd-lambda 0.25 "${COMMON[@]}"

  run_paper_train "${PREFIX}-guided-w1-s1" 100 \
    --manifest "$DENSE" --kd-mode guided \
    --temperature 1 --kd-weight 1 "${COMMON[@]}"

  run_paper_train "${PREFIX}-sctc-l1-s1" 80 \
    --manifest "$SCTC" --kd-mode sctc --kd-lambda 1 "${COMMON[@]}"
  local sctc_ckpt
  sctc_ckpt=$(best_ckpt "${PREFIX}-sctc-l1-s1")
  [[ -n "$sctc_ckpt" ]] || { echo "missing S-CTC checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-sctc-ctcft-s1" 20 \
    --config "$CONFIG" --manifest "$PLAIN" --kd-mode none \
    --train-batch-size 64 --accumulate-grad-batches 8 \
    --init-ckpt "$sctc_ckpt" --lr 0.005 --warmup-steps 100

  run_paper_train "${PREFIX}-fpkd-dfkd-s1" 10 \
    --manifest "$DENSE" --kd-mode fpkd_dfkd \
    --fpkd-temp 1 "${COMMON[@]}"
  local dfkd_ckpt
  dfkd_ckpt=$(latest_last_ckpt "${PREFIX}-fpkd-dfkd-s1")
  [[ -n "$dfkd_ckpt" ]] || { echo "missing FPKD DFKD checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-fpkd-frkd-s1" 10 \
    --manifest "$CARL" --kd-mode fpkd_frkd \
    --carl-teacher-dim 640 --init-ckpt "$dfkd_ckpt" "${COMMON[@]}"
  local frkd_ckpt
  frkd_ckpt=$(latest_last_ckpt "${PREFIX}-fpkd-frkd-s1")
  [[ -n "$frkd_ckpt" ]] || { echo "missing FPKD FRKD checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-fpkd-pkd-s1" 80 \
    --manifest "$DENSE" --kd-mode fpkd_pkd \
    --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1 \
    --init-ckpt "$frkd_ckpt" "${COMMON[@]}"

  run_paper_train "${PREFIX}-carl-feature-s1" 10 \
    --manifest "$CARL" --kd-mode carl_feature \
    --carl-teacher-dim 640 "${COMMON[@]}"
  local carl_feature_ckpt
  carl_feature_ckpt=$(latest_last_ckpt "${PREFIX}-carl-feature-s1")
  [[ -n "$carl_feature_ckpt" ]] || { echo "missing CARL feature checkpoint" >&2; exit 1; }
  run_paper_train "${PREFIX}-carl-full-s1" 90 \
    --manifest "$CARL" --kd-mode carl \
    --carl-classifier "$CLASSIFIER" --carl-teacher-dim 640 \
    --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
    --init-ckpt "$carl_feature_ckpt" "${COMMON[@]}"

  # Two augmented views per optimizer step: 50 epochs is compute-matched to 100.
  run_paper_train "${PREFIX}-crctc-fair50-s1" 50 \
    --manifest "$PLAIN" --kd-mode cr_ctc \
    --cr-ctc-weight 0.2 --cr-ctc-warm-step 1000 \
    --cr-ctc-time-factor 1.5 "${COMMON[@]}"
}

evaluate_all() {
  eval_one "${PREFIX}-no-kd" arch-citrinet144-t256-lbs-nokd-b64a8-lr025-e100-s1
  eval_one "${PREFIX}-atdk" "${PREFIX}-atdk-w12p5-ntdk4-wu1000-kdstart500-s1"
  eval_one "${PREFIX}-vanilla" "${PREFIX}-vanilla-l025-s1"
  eval_one "${PREFIX}-kdbe" "${PREFIX}-kdbe-l025-s1"
  eval_one "${PREFIX}-symmetric" "${PREFIX}-symmetric-l025-n4-s1"
  eval_one "${PREFIX}-guided" "${PREFIX}-guided-w1-s1"
  eval_one "${PREFIX}-sctc-ft" "${PREFIX}-sctc-ctcft-s1"
  eval_one "${PREFIX}-fpkd" "${PREFIX}-fpkd-pkd-s1"
  eval_one "${PREFIX}-carl" "${PREFIX}-carl-full-s1"
  eval_one "${PREFIX}-crctc" "${PREFIX}-crctc-fair50-s1"
}

case "${1:-all}" in
  ours) run_ours ;;
  prepare) prepare_full_targets ;;
  baselines) run_baselines ;;
  eval) evaluate_all ;;
  all)
    run_ours
    prepare_full_targets
    run_baselines
    evaluate_all
    ;;
  *)
    echo "usage: $0 [ours|prepare|baselines|eval|all]" >&2
    exit 2
    ;;
esac
