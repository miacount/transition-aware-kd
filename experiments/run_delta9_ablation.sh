#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

BEAM_WIDTH="${BEAM_WIDTH:-16}"
SEEDS="${SEEDS:-1}"
DATASETS="${DATASETS:-lbs chime3}"
LS_D9=data/train_clean_100.teacher_only_d9_mass3_ntdk_m32.json
CH_D9=data/chime3_train.adapted_teacher.d9_mass3_ntdk_m32.json
mkdir -p analysis

for dataset in $DATASETS; do
  case "$dataset" in
    lbs) require_file "$LS_D9" ;;
    chime3) require_file "$CH_D9" ;;
    *) echo "unknown DATASETS entry: $dataset" >&2; exit 2 ;;
  esac
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

evaluate_lbs() {
  local seed="$1" name="ablation-lbs-full-d9-mass25-ntdk8-s${seed}" ckpt log
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  log="analysis/beam${BEAM_WIDTH}_ablation_lbs_full_d9_s${seed}.txt"
  if [[ -s "$log" ]] && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" --name "ablation_lbs_full_d9_s${seed}" \
    --config configs/student_base.yaml --beam-width "$BEAM_WIDTH" \
    --manifest dev_clean=data/dev_clean.json \
    --manifest dev_other=data/dev_other.json \
    --manifest test_clean=data/test_clean.json \
    --manifest test_other=data/test_other.json | tee "$log"
}

evaluate_chime3() {
  local seed="$1" name="ablation-chime3-full-d9-mass25-ntdk8-lr05-s${seed}" ckpt log
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  log="analysis/beam${BEAM_WIDTH}_ablation_chime3_full_d9_s${seed}.txt"
  if [[ -s "$log" ]] && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" --name "ablation_chime3_full_d9_s${seed}" \
    --config configs/student_base_chime3_full.yaml --beam-width "$BEAM_WIDTH" \
    --manifest chime3_dev_real=data/chime3_dev_real_enhanced.json \
    --manifest chime3_dev_simu=data/chime3_dev_simu_enhanced.json \
    --manifest chime3_eval_real=data/chime3_eval_real_enhanced.json \
    --manifest chime3_eval_simu=data/chime3_eval_simu_enhanced.json | tee "$log"
}

for seed in $SEEDS; do
  case "$seed" in
    1|2|3) ;;
    *) echo "SEEDS must contain only 1, 2, or 3" >&2; exit 2 ;;
  esac
  for dataset in $DATASETS; do
    if [[ "$dataset" == lbs ]]; then
      name="ablation-lbs-full-d9-mass25-ntdk8-s${seed}"
      run_seeded_train "$name" 100 "$seed" \
        --config student_base --manifest "$LS_D9" --kd-mode span_kd \
        --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8
      evaluate_lbs "$seed"
    else
      name="ablation-chime3-full-d9-mass25-ntdk8-lr05-s${seed}"
      if ! experiment_complete "$name" 100 && ! experiment_complete "$name" 10; then
        run_seeded_train "$name" 10 "$seed" \
          --config student_base_chime3_runtime --manifest "$CH_D9" --kd-mode span_kd \
          --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8 \
          --lr 0.5 --warmup-steps 750 --dev-only
        smoke_ckpt=$(best_ckpt "$name")
        smoke_wer=$(basename "$smoke_ckpt" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
        awk -v wer="$smoke_wer" 'BEGIN { exit !(wer >= 0 && wer < 0.95) }' || {
          echo "collapsed CHIME-3 smoke run: $name best val_wer=$smoke_wer" >&2
          exit 1
        }
      fi
      run_seeded_train "$name" 100 "$seed" \
        --config student_base_chime3_runtime --manifest "$CH_D9" --kd-mode span_kd \
        --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8 \
        --lr 0.5 --warmup-steps 750 --dev-only
      evaluate_chime3 "$seed"
    fi
  done
done

if [[ "$SEEDS" == "1" && "$DATASETS" == "lbs chime3" ]]; then
  python scripts/summarize_delta9_ablation.py --beam-width "$BEAM_WIDTH"
fi
echo "=== delta=9 ablation complete ==="
