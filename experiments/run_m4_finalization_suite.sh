#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

BEAM_WIDTH="${BEAM_WIDTH:-16}"
mkdir -p analysis

LS_D0=data/train_clean_100.teacher_only_d0_mass3_ntdk_m4.json
LS_D6=data/train_clean_100.teacher_only_d6_mass3_ntdk_m4.json
LS_D9=data/train_clean_100.teacher_only_d9_mass3_ntdk_m4.json
LS_PERM=data/train_clean_100.teacher_only_d6_mass3_ntdk_m4_permuted.json
CH_D0=data/chime3_train.adapted_teacher.d0_mass3_ntdk_m4.json
CH_D6=data/chime3_train.adapted_teacher.d6_mass3_ntdk_m4.json
CH_D9=data/chime3_train.adapted_teacher.d9_mass3_ntdk_m4.json
for path in "$LS_D0" "$LS_D6" "$LS_D9" "$LS_PERM" "$CH_D0" "$CH_D6" "$CH_D9"; do
  require_file "$path"
done

run_seeded_train() {
  local name="$1" epochs="$2" seed="$3"
  shift 3
  if experiment_complete "$name" "$epochs"; then
    echo "=== [reuse] $name seed=$seed epochs=$epochs ==="
    return 0
  fi
  local last version
  last=$(latest_last_ckpt "$name")
  if [[ -n "$last" ]]; then
    version=$(basename "$(dirname "$(dirname "$last")")")
    echo "=== [resume] $name seed=$seed from $last ==="
    bash experiments/train.sh --name "$name" --epochs "$epochs" --seed "$seed" \
      --resume-version "$version" "$@"
  else
    echo "=== [train] $name seed=$seed epochs=$epochs ==="
    bash experiments/train.sh --name "$name" --epochs "$epochs" --seed "$seed" "$@"
  fi
  [[ -n "$(best_ckpt "$name")" ]] || {
    echo "no finite checkpoint after training: $name" >&2
    exit 1
  }
}

run_chime_guarded() {
  local name="$1" seed="$2"
  shift 2
  if ! experiment_complete "$name" 100; then
    local checkpoint wer last last_epoch
    last=$(latest_last_ckpt "$name")
    last_epoch=$(basename "$last" 2>/dev/null | sed -n 's/.*epoch=\([0-9]*\)-last.*/\1/p')
    if [[ -z "$last_epoch" || "$last_epoch" -lt 10 ]]; then
      run_seeded_train "$name" 10 "$seed" "$@"
    fi
    checkpoint=$(best_ckpt "$name")
    wer=$(basename "$checkpoint" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
    awk -v wer="$wer" 'BEGIN { exit !(wer >= 0 && wer < 0.95) }' || {
      echo "collapsed CHIME smoke: $name best val_wer=$wer" >&2
      exit 1
    }
  fi
  run_seeded_train "$name" 100 "$seed" "$@"
}

evaluate_one() {
  local dataset="$1" tag="$2" seed="$3" name="$4" ckpt log
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  log="analysis/beam${BEAM_WIDTH}_m4final_${dataset}_${tag}_s${seed}.txt"
  if [[ -s "$log" ]] && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  echo "=== [beam${BEAM_WIDTH}] $dataset/$tag seed=$seed ==="
  if [[ "$dataset" == lbs ]]; then
    python scripts/evaluate_ctc_decoding.py \
      --ckpt "$ckpt" --name "m4final_${dataset}_${tag}_s${seed}" \
      --config configs/student_base.yaml --beam-width "$BEAM_WIDTH" \
      --manifest dev_clean=data/dev_clean.json \
      --manifest dev_other=data/dev_other.json \
      --manifest test_clean=data/test_clean.json \
      --manifest test_other=data/test_other.json | tee "$log"
  else
    python scripts/evaluate_ctc_decoding.py \
      --ckpt "$ckpt" --name "m4final_${dataset}_${tag}_s${seed}" \
      --config configs/student_base_chime3_full.yaml --beam-width "$BEAM_WIDTH" \
      --manifest chime3_dev_real=data/chime3_dev_real_enhanced.json \
      --manifest chime3_dev_simu=data/chime3_dev_simu_enhanced.json \
      --manifest chime3_eval_real=data/chime3_eval_real_enhanced.json \
      --manifest chime3_eval_simu=data/chime3_eval_simu_enhanced.json | tee "$log"
  fi
}

run_lbs() {
  local tag="$1" name="$2" seed="$3" manifest="$4" mass_w="$5" ntdk_w="$6"
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$manifest" --kd-mode span_kd \
    --kd-weight "$mass_w" --span-primary-mode mass3 --span-ntdk-weight "$ntdk_w"
  evaluate_one lbs "$tag" "$seed" "$name"
}

run_chime() {
  local tag="$1" name="$2" seed="$3" manifest="$4" mass_w="$5" ntdk_w="$6"
  run_chime_guarded "$name" "$seed" \
    --config student_base_chime3_runtime --manifest "$manifest" --kd-mode span_kd \
    --kd-weight "$mass_w" --span-primary-mode mass3 --span-ntdk-weight "$ntdk_w" \
    --lr 0.5 --warmup-steps 750 --dev-only
  evaluate_one chime3 "$tag" "$seed" "$name"
}

echo "========== PHASE 1: M=4 MATCHED THREE-SEED MAIN METHOD =========="
for seed in 1 2 3; do
  if [[ "$seed" == 1 ]]; then
    name=ablation-lbs-ntdk-topm4-mass25-w8-s1
  else
    name="paper-lbs-main3-m4-ours-s${seed}"
  fi
  run_lbs full_d6 "$name" "$seed" "$LS_D6" 25 8
done
for seed in 1 2 3; do
  run_chime full_d6 "paper-chime3-main3-m4-ours-s${seed}" "$seed" "$CH_D6" 25 8
done

echo "========== PHASE 2: M=4 FACTORIAL/SEMANTIC CONTROLS =========="
run_lbs ntdk_only ablation-lbs-m4-ntdk-only-w8-s1 1 "$LS_D6" 0 8
run_chime ntdk_only ablation-chime3-m4-ntdk-only-w8-lr05-s1 1 "$CH_D6" 0 8
run_lbs full_d0 ablation-lbs-m4-full-d0-mass25-ntdk8-s1 1 "$LS_D0" 25 8
run_chime full_d0 ablation-chime3-m4-full-d0-mass25-ntdk8-lr05-s1 1 "$CH_D0" 25 8
run_lbs full_d6_permuted ablation-lbs-m4-full-d6-permuted-ntdk8-s1 1 "$LS_PERM" 25 8

echo "========== PHASE 3: M=4 DELTA=9 SENSITIVITY =========="
run_lbs full_d9 ablation-lbs-m4-full-d9-mass25-ntdk8-s1 1 "$LS_D9" 25 8
run_chime full_d9 ablation-chime3-m4-full-d9-mass25-ntdk8-lr05-s1 1 "$CH_D9" 25 8

python scripts/summarize_m4_finalization.py --beam-width "$BEAM_WIDTH"
echo "=== M=4 finalization suite complete ==="
