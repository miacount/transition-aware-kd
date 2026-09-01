#!/usr/bin/env bash
set -euo pipefail

# Actual paper main-table suite: fixed Teacher reference plus three matched
# student seeds. All reported scores use no-LM beam-16; AT-DKD uses M=32.
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

BEAM_WIDTH="${BEAM_WIDTH:-16}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p analysis

LBS_PLAIN=data/train_clean_100.json
LBS_DENSE=data/train_clean_100.small_teacher.frame_dense_t1.json
LBS_SCTC=data/train_clean_100.sctc.json
LBS_CARL=data/train_clean_100.small_teacher.carl.json
LBS_CLASSIFIER=data/carl_lbs_small/teacher_classifier.pt
LBS_M32=data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json
CH_PLAIN=data/chime3_train_enhanced.json
CH_DENSE=data/chime3_train.adapted_teacher.frame_dense_t1.json
CH_SCTC=data/chime3_train.adapted_teacher.sctc.json
CH_CARL=data/chime3_train.adapted_teacher.carl.json
CH_CLASSIFIER=data/carl_chime3_adapted/teacher_classifier.pt
CH_M32=data/chime3_train.adapted_teacher.d6_mass3_ntdk_m32.json

for path in \
  "$LBS_PLAIN" "$LBS_DENSE" "$LBS_SCTC" "$LBS_CARL" "$LBS_CLASSIFIER" "$LBS_M32" \
  "$CH_PLAIN" "$CH_DENSE" "$CH_SCTC" "$CH_CARL" "$CH_CLASSIFIER" "$CH_M32" \
  analysis/beam16_teacher_lbs_chime3.txt; do
  require_file "$path"
done

experiment_name() {
  local dataset="$1" stage="$2" seed="$3"
  if [[ "$seed" == 1 ]]; then
    case "$dataset:$stage" in
      lbs:nokd) echo student-no-kd ;;
      lbs:vanilla) echo paper-lbs-vanilla-full-l025-s1 ;;
      lbs:symmetric) echo paper-lbs-symmetric-full-l025-n4-s1 ;;
      lbs:guided) echo paper-lbs-guided-exact-w1-s1 ;;
      lbs:sctc) echo paper-lbs-sctc-l1-s1 ;;
      lbs:sctc_ft) echo paper-lbs-sctc-l1-ctcft-s1 ;;
      lbs:fpkd_dfkd) echo paper-lbs-fpkd-stablekl-dfkd-e10-wu1000-s1 ;;
      lbs:fpkd_frkd) echo paper-lbs-fpkd-stablekl-frkd-e10-wu1000-s1 ;;
      lbs:fpkd) echo paper-lbs-fpkd-stablekl-pkd-e80-wu8000-bkl1-nbf1-t1-s1 ;;
      lbs:carl_feature) echo paper-lbs-carl-feature-e10-wu1000-s1 ;;
      lbs:carl) echo paper-lbs-carl-full-ctc-e50-wu5000-a1-g1-l1-s1 ;;
      lbs:crctc) echo cr_ctc_w02_fair50 ;;
      lbs:ours) echo span-d6-mass3-w25-ntdk-a8-s1 ;;
      chime3:nokd) echo paper-chime3-lr05-no-kd-s1 ;;
      chime3:vanilla) echo paper-chime3-vanilla-full-l025-lr05-s1 ;;
      chime3:symmetric) echo paper-chime3-symmetric-full-l025-n4-lr05-s1 ;;
      chime3:guided) echo paper-chime3-guided-exact-w1-lr05-s1 ;;
      chime3:sctc) echo paper-chime3-sctc-l1-lr05-s1 ;;
      chime3:sctc_ft) echo paper-chime3-sctc-l1-ctcft-lr005-wu100-s1 ;;
      chime3:fpkd_dfkd) echo paper-chime3-fpkd-stablekl-dfkd-e10-wu300-lr05-s1 ;;
      chime3:fpkd_frkd) echo paper-chime3-fpkd-stablekl-frkd-e10-wu300-lr05-s1 ;;
      chime3:fpkd) echo paper-chime3-fpkd-stablekl-pkd-e80-wu2500-lr05-bkl1-nbf1-t1-s1 ;;
      chime3:carl_feature) echo paper-chime3-carl-feature-e10-wu300-lr05-s1 ;;
      chime3:carl) echo paper-chime3-carl-full-ctc-e50-wu1500-lr05-a1-g1-l1-s1 ;;
      chime3:crctc) echo paper-chime3-crctc-nemo15-e50-lr05-wu750-s1 ;;
      chime3:ours) echo paper-chime3-lr05-mass3-w25-ntdk8-s1 ;;
      *) echo "unknown dataset/stage: $dataset/$stage" >&2; return 2 ;;
    esac
  else
    echo "paper-${dataset}-main3-m32-${stage}-s${seed}"
  fi
}

best_ckpt_upto() {
  local name="$1" max_epoch="$2"
  find "nemo_experiments/$name" -type f -name '*.ckpt' ! -name '*last.ckpt' -print 2>/dev/null \
    | while read -r file; do
        local base wer epoch
        base=$(basename "$file")
        wer=$(sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p' <<< "$base")
        epoch=$(sed -n 's/.*epoch=\([0-9]*\).*/\1/p' <<< "$base")
        if [[ -n "$wer" && -n "$epoch" && "$epoch" -le "$max_epoch" ]]; then
          printf '%s %s\n' "$wer" "$file"
        fi
      done | sort -n | head -1 | cut -d' ' -f2-
}

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
    echo "no finite checkpoint after training: $name" >&2; exit 1;
  }
}

run_chime_guarded() {
  local name="$1" epochs="$2" smoke_epochs="$3" seed="$4"
  shift 4
  if experiment_complete "$name" "$epochs"; then
    echo "=== [reuse] $name seed=$seed epochs=$epochs ==="
    return 0
  fi
  if ! experiment_complete "$name" "$smoke_epochs"; then
    run_seeded_train "$name" "$smoke_epochs" "$seed" "$@" --dev-only
  fi
  local checkpoint wer
  checkpoint=$(best_ckpt "$name")
  [[ -n "$checkpoint" ]] || { echo "CHIME smoke produced no checkpoint: $name" >&2; exit 1; }
  wer=$(basename "$checkpoint" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
  awk -v wer="$wer" 'BEGIN { exit !(wer >= 0 && wer < 0.95) }' || {
    echo "collapsed CHIME smoke: $name best val_wer=$wer" >&2; exit 1;
  }
  run_seeded_train "$name" "$epochs" "$seed" "$@" --dev-only
}

run_chime_stage() {
  local name="$1" epochs="$2" seed="$3"
  shift 3
  # Feature-only stages are not decodable ASR models; their WER may be 1.0
  # or higher until the later output or CTC stage.
  run_seeded_train "$name" "$epochs" "$seed" "$@" --dev-only
}

train_lbs_seed() {
  local seed="$1" name ckpt
  name=$(experiment_name lbs nokd "$seed")
  run_seeded_train "$name" 100 "$seed" --config student_base --manifest "$LBS_PLAIN" --kd-mode none
  name=$(experiment_name lbs vanilla "$seed")
  run_seeded_train "$name" 100 "$seed" --config student_base --manifest "$LBS_DENSE" \
    --kd-mode logit --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25
  name=$(experiment_name lbs symmetric "$seed")
  run_seeded_train "$name" 100 "$seed" --config student_base --manifest "$LBS_DENSE" \
    --kd-mode logit --temperature 1 --logit-reduction utterance_sum \
    --blank-mode symmetric --blank-n 4 --kd-lambda 0.25
  name=$(experiment_name lbs guided "$seed")
  run_seeded_train "$name" 100 "$seed" --config student_base --manifest "$LBS_DENSE" \
    --kd-mode guided --temperature 1 --kd-weight 1

  name=$(experiment_name lbs sctc "$seed")
  run_seeded_train "$name" 80 "$seed" --config student_base --manifest "$LBS_SCTC" --kd-mode sctc --kd-lambda 1
  ckpt=$(best_ckpt "$name")
  name=$(experiment_name lbs sctc_ft "$seed")
  run_seeded_train "$name" 20 "$seed" --config student_base --manifest "$LBS_PLAIN" \
    --kd-mode none --init-ckpt "$ckpt"

  name=$(experiment_name lbs fpkd_dfkd "$seed")
  run_seeded_train "$name" 10 "$seed" --config student_base --manifest "$LBS_DENSE" \
    --kd-mode fpkd_dfkd --fpkd-temp 1 --warmup-steps 1000
  ckpt=$(latest_last_ckpt "$name")
  name=$(experiment_name lbs fpkd_frkd "$seed")
  run_seeded_train "$name" 10 "$seed" --config student_base --manifest "$LBS_CARL" \
    --kd-mode fpkd_frkd --warmup-steps 1000 --init-ckpt "$ckpt"
  ckpt=$(latest_last_ckpt "$name")
  name=$(experiment_name lbs fpkd "$seed")
  run_seeded_train "$name" 80 "$seed" --config student_base --manifest "$LBS_DENSE" \
    --kd-mode fpkd_pkd --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1 \
    --warmup-steps 8000 --init-ckpt "$ckpt"

  name=$(experiment_name lbs carl_feature "$seed")
  run_seeded_train "$name" 10 "$seed" --config student_base --manifest "$LBS_CARL" \
    --kd-mode carl_feature --carl-teacher-dim 176 --warmup-steps 1000
  ckpt=$(latest_last_ckpt "$name")
  name=$(experiment_name lbs carl "$seed")
  if [[ "$seed" == 1 ]]; then
    [[ -n "$(best_ckpt_upto "$name" 50)" ]] || { echo "missing seed-1 LBS CARL <=50 checkpoint" >&2; exit 1; }
    echo "=== [reuse] $name seed=1 original 10+50 budget ==="
  else
    run_seeded_train "$name" 50 "$seed" --config student_base --manifest "$LBS_CARL" \
      --kd-mode carl --carl-classifier "$LBS_CLASSIFIER" --carl-teacher-dim 176 \
      --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 --warmup-steps 5000 \
      --save-top-k -1 --init-ckpt "$ckpt"
  fi

  name=$(experiment_name lbs crctc "$seed")
  run_seeded_train "$name" 50 "$seed" --config student_base --manifest "$LBS_PLAIN" \
    --kd-mode cr_ctc --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 --cr-ctc-time-factor 1.5
  name=$(experiment_name lbs ours "$seed")
  run_seeded_train "$name" 100 "$seed" --config student_base --manifest "$LBS_M32" \
    --kd-mode span_kd --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8
}

train_chime_seed() {
  local seed="$1" name ckpt
  local common=(--config student_base_chime3_full --lr 0.5 --warmup-steps 750)
  name=$(experiment_name chime3 nokd "$seed")
  run_chime_guarded "$name" 100 10 "$seed" --config student_base_chime3_runtime \
    --manifest "$CH_PLAIN" --kd-mode none --lr 0.5 --warmup-steps 750
  name=$(experiment_name chime3 vanilla "$seed")
  run_chime_guarded "$name" 100 10 "$seed" "${common[@]}" --manifest "$CH_DENSE" \
    --kd-mode logit --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25
  name=$(experiment_name chime3 symmetric "$seed")
  run_chime_guarded "$name" 100 10 "$seed" "${common[@]}" --manifest "$CH_DENSE" \
    --kd-mode logit --temperature 1 --logit-reduction utterance_sum \
    --blank-mode symmetric --blank-n 4 --kd-lambda 0.25
  name=$(experiment_name chime3 guided "$seed")
  run_chime_guarded "$name" 100 10 "$seed" "${common[@]}" --manifest "$CH_DENSE" \
    --kd-mode guided --temperature 1 --kd-weight 1

  name=$(experiment_name chime3 sctc "$seed")
  run_chime_guarded "$name" 80 10 "$seed" "${common[@]}" --manifest "$CH_SCTC" --kd-mode sctc --kd-lambda 1
  ckpt=$(best_ckpt "$name")
  name=$(experiment_name chime3 sctc_ft "$seed")
  run_chime_guarded "$name" 20 3 "$seed" --config student_base_chime3_full \
    --manifest "$CH_PLAIN" --kd-mode none --init-ckpt "$ckpt" --lr 0.05 --warmup-steps 100

  name=$(experiment_name chime3 fpkd_dfkd "$seed")
  run_chime_stage "$name" 10 "$seed" --config student_base_chime3_full \
    --manifest "$CH_DENSE" --kd-mode fpkd_dfkd --fpkd-temp 1 --lr 0.5 --warmup-steps 300
  ckpt=$(latest_last_ckpt "$name")
  name=$(experiment_name chime3 fpkd_frkd "$seed")
  run_chime_stage "$name" 10 "$seed" --config student_base_chime3_full \
    --manifest "$CH_CARL" --kd-mode fpkd_frkd --init-ckpt "$ckpt" --lr 0.5 --warmup-steps 300
  ckpt=$(latest_last_ckpt "$name")
  name=$(experiment_name chime3 fpkd "$seed")
  run_chime_stage "$name" 80 "$seed" --config student_base_chime3_full \
    --manifest "$CH_DENSE" --kd-mode fpkd_pkd --fpkd-bkl-weight 1 \
    --fpkd-nbf-weight 1 --fpkd-temp 1 --init-ckpt "$ckpt" --lr 0.5 --warmup-steps 2500

  name=$(experiment_name chime3 carl_feature "$seed")
  run_chime_stage "$name" 10 "$seed" --config student_base_chime3_full \
    --manifest "$CH_CARL" --kd-mode carl_feature --carl-teacher-dim 176 --lr 0.5 --warmup-steps 300
  ckpt=$(latest_last_ckpt "$name")
  name=$(experiment_name chime3 carl "$seed")
  if [[ "$seed" == 1 ]]; then
    [[ -n "$(best_ckpt_upto "$name" 50)" ]] || { echo "missing seed-1 CHIME CARL <=50 checkpoint" >&2; exit 1; }
    echo "=== [reuse] $name seed=1 original 10+50 budget ==="
  else
    run_chime_guarded "$name" 50 10 "$seed" --config student_base_chime3_full \
      --manifest "$CH_CARL" --kd-mode carl --carl-classifier "$CH_CLASSIFIER" \
      --carl-teacher-dim 176 --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
      --init-ckpt "$ckpt" --lr 0.5 --warmup-steps 1500 --save-top-k -1
  fi

  name=$(experiment_name chime3 crctc "$seed")
  run_chime_guarded "$name" 50 10 "$seed" "${common[@]}" --manifest "$CH_PLAIN" \
    --kd-mode cr_ctc --cr-ctc-weight 0.2 --cr-ctc-warm-step 750 \
    --cr-ctc-time-factor 1.5 --train-batch-size 32 --accumulate-grad-batches 2
  name=$(experiment_name chime3 ours "$seed")
  run_chime_guarded "$name" 100 10 "$seed" --config student_base_chime3_runtime \
    --manifest "$CH_M32" --kd-mode span_kd --kd-weight 25 --span-primary-mode mass3 \
    --span-ntdk-weight 8 --lr 0.5 --warmup-steps 750
}

method_checkpoint() {
  local dataset="$1" method="$2" seed="$3" name
  name=$(experiment_name "$dataset" "$method" "$seed")
  if [[ "$method" == carl ]]; then best_ckpt_upto "$name" 50; else best_ckpt "$name"; fi
}

evaluate_one() {
  local dataset="$1" method="$2" seed="$3" ckpt log
  ckpt=$(method_checkpoint "$dataset" "$method" "$seed")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $dataset/$method/seed$seed" >&2; exit 1; }
  log="analysis/beam${BEAM_WIDTH}_main3seed_${dataset}_${method}_s${seed}.txt"
  if [[ -s "$log" ]] && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  if [[ "$dataset" == lbs ]]; then
    python scripts/evaluate_ctc_decoding.py --ckpt "$ckpt" --name "${dataset}_${method}_s${seed}" \
      --config configs/student_base.yaml --beam-width "$BEAM_WIDTH" \
      --manifest dev_clean=data/dev_clean.json --manifest dev_other=data/dev_other.json \
      --manifest test_clean=data/test_clean.json --manifest test_other=data/test_other.json | tee "$log"
  else
    python scripts/evaluate_ctc_decoding.py --ckpt "$ckpt" --name "${dataset}_${method}_s${seed}" \
      --config configs/student_base_chime3_full.yaml --beam-width "$BEAM_WIDTH" \
      --manifest chime3_dev_real=data/chime3_dev_real_enhanced.json \
      --manifest chime3_dev_simu=data/chime3_dev_simu_enhanced.json \
      --manifest chime3_eval_real=data/chime3_eval_real_enhanced.json \
      --manifest chime3_eval_simu=data/chime3_eval_simu_enhanced.json | tee "$log"
  fi
}

METHODS="nokd vanilla symmetric guided sctc_ft fpkd carl crctc ours"
for seed in $SEEDS; do
  case "$seed" in 1|2|3) ;; *) echo "SEEDS must contain only 1, 2, 3" >&2; exit 2 ;; esac
  echo "========== LIBRISPEECH MAIN TABLE SEED $seed =========="
  train_lbs_seed "$seed"
  for method in $METHODS; do evaluate_one lbs "$method" "$seed"; done
done
for seed in $SEEDS; do
  echo "========== CHIME-3 MAIN TABLE SEED $seed =========="
  train_chime_seed "$seed"
  for method in $METHODS; do evaluate_one chime3 "$method" "$seed"; done
done

python scripts/summarize_main_table_3seed.py --beam-width "$BEAM_WIDTH"
echo "=== M=32 main table 3-seed suite complete ==="
