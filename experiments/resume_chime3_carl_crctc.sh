#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

CONFIG=student_base_chime3_full
PLAIN=data/chime3_train_enhanced.json
CARL=data/chime3_train.adapted_teacher.carl.json
CLASSIFIER=data/carl_chime3_adapted/teacher_classifier.pt
CARL_FEATURE=paper-chime3-carl-feature-e10-wu300-lr05-s1
CARL_FULL=paper-chime3-carl-full-ctc-e50-wu1500-lr05-a1-g1-l1-s1
CR_NAME=paper-chime3-crctc-nemo15-e50-lr05-wu750-s1

for path in "$PLAIN" "$CARL" "$CLASSIFIER"; do require_file "$path"; done
export EVAL_MANIFESTS="chime3_dev_real=data/chime3_dev_real_enhanced.json chime3_dev_simu=data/chime3_dev_simu_enhanced.json chime3_eval_real=data/chime3_eval_real_enhanced.json chime3_eval_simu=data/chime3_eval_simu_enhanced.json"

last_completed_epochs() {
  local checkpoint epoch
  checkpoint=$(latest_last_ckpt "$1")
  [[ -n "$checkpoint" ]] || { echo 0; return; }
  epoch=$(basename "$checkpoint" | sed -n 's/.*epoch=\([0-9][0-9]*\)-last.*/\1/p')
  echo "${epoch:-0}"
}

gate_asr() {
  local name="$1" max_wer="$2" checkpoint wer
  checkpoint=$(best_ckpt "$name")
  [[ -n "$checkpoint" ]] || { echo "[gate] no finite checkpoint: $name" >&2; exit 1; }
  wer=$(basename "$checkpoint" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
  awk -v wer="$wer" -v max="$max_wer" 'BEGIN { exit !(wer >= 0 && wer < max) }' || {
    echo "[gate] ASR did not recover: $name best val_wer=$wer (required <$max_wer)" >&2
    exit 1
  }
  echo "[gate] $name passed: best val_wer=$wer"
}

# CARL feature-only pretraining has no meaningful ASR decoder. Continue the
# existing epoch-3 checkpoint to epoch 10 and validate tensors, not WER.
if ! experiment_complete "$CARL_FEATURE" 10; then
  run_paper_train "$CARL_FEATURE" 10 \
    --config "$CONFIG" --manifest "$CARL" --kd-mode carl_feature \
    --carl-teacher-dim 176 --lr 0.5 --warmup-steps 300
fi
CARL_FEATURE_CKPT=$(latest_last_ckpt "$CARL_FEATURE")
[[ -n "$CARL_FEATURE_CKPT" ]] || { echo "missing CARL feature checkpoint" >&2; exit 1; }
python scripts/check_checkpoint_finite.py "$CARL_FEATURE_CKPT"

# The full CARL stage starts from a feature-only model and uses an 11-epoch
# Noam warm-up. Gate at epoch 20, after the decoder has had time to recover.
CARL_ARGS=(
  --config "$CONFIG" --manifest "$CARL" --kd-mode carl
  --carl-classifier "$CLASSIFIER" --carl-teacher-dim 176
  --carl-alpha 1 --carl-gamma 1 --carl-lambda 1
  --init-ckpt "$CARL_FEATURE_CKPT" --lr 0.5 --warmup-steps 1500 --save-top-k -1
)
if ! experiment_complete "$CARL_FULL" 50; then
  completed=$(last_completed_epochs "$CARL_FULL")
  if (( completed < 20 )); then
    run_paper_train "$CARL_FULL" 20 "${CARL_ARGS[@]}"
  fi
  gate_asr "$CARL_FULL" 0.95
  run_paper_train "$CARL_FULL" 50 "${CARL_ARGS[@]}"
fi

CARL_AVG="nemo_experiments/$CARL_FULL/carl-last10-avg.ckpt"
python scripts/average_checkpoints.py \
  --experiment-dir "nemo_experiments/$CARL_FULL" --last-n 10 --output "$CARL_AVG"
python scripts/check_checkpoint_finite.py "$CARL_AVG"

# CR-CTC uses the 5.5-epoch shared warm-up. Gate at epoch 12, then continue to
# the compute-matched 50 dual-view epochs.
CR_ARGS=(
  --config "$CONFIG" --manifest "$PLAIN" --kd-mode cr_ctc
  --lr 0.5 --warmup-steps 750
  --cr-ctc-weight 0.2 --cr-ctc-warm-step 750 --cr-ctc-time-factor 1.5
  --train-batch-size 32 --accumulate-grad-batches 2
)
if ! experiment_complete "$CR_NAME" 50; then
  completed=$(last_completed_epochs "$CR_NAME")
  if (( completed < 12 )); then
    run_paper_train "$CR_NAME" 12 "${CR_ARGS[@]}"
  fi
  gate_asr "$CR_NAME" 0.95
  run_paper_train "$CR_NAME" 50 "${CR_ARGS[@]}"
fi

bash experiments/evaluate_chime3_baseline_suite_lr05.sh
echo "[done] CARL, CR-CTC, and final Beam-16 evaluation"
