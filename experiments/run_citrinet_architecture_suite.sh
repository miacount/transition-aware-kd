#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

STAGE="${1:-pilot}"
SCOPE="${2:-all}"
case "$STAGE" in pilot|full|eval) ;; *) echo "usage: $0 [pilot|full|eval] [lbs|chime3|all]" >&2; exit 2 ;; esac
case "$SCOPE" in lbs|chime3|all) ;; *) echo "usage: $0 [pilot|full|eval] [lbs|chime3|all]" >&2; exit 2 ;; esac

run_eval() {
  local label="$1" name="$2" config="$3"; shift 3
  local ckpt
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  python scripts/evaluate_ctc_decoding.py --ckpt "$ckpt" --name "$label"     --config "$config" --beam-width 16 "$@" | tee "analysis/beam16_${label}.txt"
}

run_pilot() {
  local scope="$1" config plain mass warmup prefix
  if [[ "$scope" == lbs ]]; then
    config=student_citrinet256
    plain=data/train_clean_100.json
    mass=data/train_clean_100.citrinet512.d6_mass3_ntdk_m32.json
    warmup=5000
    prefix=arch-citrinet-lbs
  else
    config=student_citrinet256_chime3
    plain=data/chime3_train_enhanced.json
    mass=data/chime3_train.citrinet512_adapted.d6_mass3_ntdk_m32.json
    warmup=300
    prefix=arch-citrinet-chime3
  fi
  require_file "$mass"
  run_paper_train "${prefix}-no-kd-s1" 100     --config "$config" --manifest "$plain" --kd-mode none --warmup-steps "$warmup"
  run_paper_train "${prefix}-at-dkd-s1" 100     --config "$config" --manifest "$mass" --kd-mode span_kd     --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8     --warmup-steps "$warmup"
}

run_full() {
  local scope="$1" config plain dense sctc carl classifier warmup prefix
  if [[ "$scope" == lbs ]]; then
    config=student_citrinet256
    plain=data/train_clean_100.json
    dense=data/train_clean_100.citrinet512.frame_dense_t1.json
    sctc=data/train_clean_100.citrinet512.sctc.json
    carl=data/train_clean_100.citrinet512.carl.json
    classifier=data/train_clean_100.citrinet512.carl/teacher_classifier.pt
    warmup=1000
    prefix=arch-citrinet-lbs
  else
    config=student_citrinet256_chime3
    plain=data/chime3_train_enhanced.json
    dense=data/chime3_train.citrinet512_adapted.frame_dense_t1.json
    sctc=data/chime3_train.citrinet512_adapted.sctc.json
    carl=data/chime3_train.citrinet512_adapted.carl.json
    classifier=data/chime3_train.citrinet512_adapted.carl/teacher_classifier.pt
    warmup=300
    prefix=arch-citrinet-chime3
  fi
  for f in "$dense" "$sctc" "$carl" "$classifier"; do require_file "$f"; done
  run_pilot "$scope"

  run_paper_train "${prefix}-vanilla-l025-s1" 100     --config "$config" --manifest "$dense" --kd-mode logit     --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25     --warmup-steps "$warmup"
  run_paper_train "${prefix}-symmetric-l025-n4-s1" 100     --config "$config" --manifest "$dense" --kd-mode logit     --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric --blank-n 4     --kd-lambda 0.25 --warmup-steps "$warmup"
  run_paper_train "${prefix}-guided-w1-s1" 100     --config "$config" --manifest "$dense" --kd-mode guided     --temperature 1 --kd-weight 1 --warmup-steps "$warmup"

  run_paper_train "${prefix}-sctc-l1-s1" 80     --config "$config" --manifest "$sctc" --kd-mode sctc --kd-lambda 1     --warmup-steps "$warmup"
  local sctc_ckpt
  sctc_ckpt=$(best_ckpt "${prefix}-sctc-l1-s1")
  run_paper_train "${prefix}-sctc-ctcft-s1" 20     --config "$config" --manifest "$plain" --kd-mode none     --init-ckpt "$sctc_ckpt" --lr 0.005 --warmup-steps 100

  run_paper_train "${prefix}-fpkd-dfkd-s1" 10     --config "$config" --manifest "$dense" --kd-mode fpkd_dfkd     --fpkd-temp 1 --warmup-steps "$warmup"
  local dfkd frkd
  dfkd=$(latest_last_ckpt "${prefix}-fpkd-dfkd-s1")
  run_paper_train "${prefix}-fpkd-frkd-s1" 10     --config "$config" --manifest "$carl" --kd-mode fpkd_frkd     --carl-teacher-dim 640 --init-ckpt "$dfkd" --warmup-steps "$warmup"
  frkd=$(latest_last_ckpt "${prefix}-fpkd-frkd-s1")
  run_paper_train "${prefix}-fpkd-pkd-s1" 80     --config "$config" --manifest "$dense" --kd-mode fpkd_pkd     --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1     --init-ckpt "$frkd" --warmup-steps "$warmup"

  run_paper_train "${prefix}-carl-feature-s1" 10     --config "$config" --manifest "$carl" --kd-mode carl_feature     --carl-teacher-dim 640 --warmup-steps "$warmup"
  local carl_feature
  carl_feature=$(latest_last_ckpt "${prefix}-carl-feature-s1")
  run_paper_train "${prefix}-carl-full-s1" 50     --config "$config" --manifest "$carl" --kd-mode carl     --carl-classifier "$classifier" --carl-teacher-dim 640     --carl-alpha 1 --carl-gamma 1 --carl-lambda 1     --init-ckpt "$carl_feature" --warmup-steps "$warmup"

  run_paper_train "${prefix}-crctc-s1" 50     --config "$config" --manifest "$plain" --kd-mode cr_ctc     --cr-ctc-weight 0.2 --cr-ctc-warm-step "$warmup"     --cr-ctc-time-factor 1.5 --warmup-steps "$warmup"
}

eval_scope() {
  local scope="$1" config prefix manifests
  if [[ "$scope" == lbs ]]; then
    config=configs/student_citrinet256.yaml
    prefix=arch-citrinet-lbs
    manifests=(--manifest dev_clean=data/dev_clean.json --manifest dev_other=data/dev_other.json --manifest test_clean=data/test_clean.json --manifest test_other=data/test_other.json)
  else
    config=configs/student_citrinet256_chime3.yaml
    prefix=arch-citrinet-chime3
    manifests=(--manifest dev_real=data/chime3_dev_real_enhanced.json --manifest dev_simu=data/chime3_dev_simu_enhanced.json --manifest eval_real=data/chime3_eval_real_enhanced.json --manifest eval_simu=data/chime3_eval_simu_enhanced.json)
  fi
  run_eval "${prefix}-no-kd" "${prefix}-no-kd-s1" "$config" "${manifests[@]}"
  run_eval "${prefix}-at-dkd" "${prefix}-at-dkd-s1" "$config" "${manifests[@]}"
  if [[ "$STAGE" == full || "${EVAL_FULL:-0}" == 1 ]]; then
    run_eval "${prefix}-vanilla" "${prefix}-vanilla-l025-s1" "$config" "${manifests[@]}"
    run_eval "${prefix}-symmetric" "${prefix}-symmetric-l025-n4-s1" "$config" "${manifests[@]}"
    run_eval "${prefix}-guided" "${prefix}-guided-w1-s1" "$config" "${manifests[@]}"
    run_eval "${prefix}-sctc-ft" "${prefix}-sctc-ctcft-s1" "$config" "${manifests[@]}"
    run_eval "${prefix}-fpkd" "${prefix}-fpkd-pkd-s1" "$config" "${manifests[@]}"
    run_eval "${prefix}-carl" "${prefix}-carl-full-s1" "$config" "${manifests[@]}"
    run_eval "${prefix}-crctc" "${prefix}-crctc-s1" "$config" "${manifests[@]}"
  fi
}

for scope in lbs chime3; do
  [[ "$SCOPE" == all || "$SCOPE" == "$scope" ]] || continue
  case "$STAGE" in
    pilot) run_pilot "$scope"; eval_scope "$scope" ;;
    full) run_full "$scope"; eval_scope "$scope" ;;
    eval) eval_scope "$scope" ;;
  esac
done
