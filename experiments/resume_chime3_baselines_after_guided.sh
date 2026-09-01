#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

NAME=paper-chime3-guided-exact-w1-lr05-s1
DENSE=data/chime3_train.adapted_teacher.frame_dense_t1.json
CONFIG=student_base_chime3_full
GATE_EPOCHS=12
FULL_EPOCHS=100
GATE_MAX_WER=0.90

require_file "$DENSE"
export EVAL_MANIFESTS="chime3_dev_real=data/chime3_dev_real_enhanced.json chime3_dev_simu=data/chime3_dev_simu_enhanced.json chime3_eval_real=data/chime3_eval_real_enhanced.json chime3_eval_simu=data/chime3_eval_simu_enhanced.json"

GUIDED_ARGS=(
  --config "$CONFIG" --manifest "$DENSE" --kd-mode guided
  --temperature 1 --kd-weight 1
  --lr 0.5 --warmup-steps 750
)

last_completed_epochs() {
  local checkpoint epoch
  checkpoint=$(latest_last_ckpt "$1")
  [[ -n "$checkpoint" ]] || { echo 0; return; }
  epoch=$(basename "$checkpoint" | sed -n 's/.*epoch=\([0-9][0-9]*\)-last.*/\1/p')
  echo "${epoch:-0}"
}

gate_guided() {
  local checkpoint wer
  checkpoint=$(best_ckpt "$NAME")
  [[ -n "$checkpoint" ]] || { echo "[guided gate] no finite checkpoint" >&2; exit 1; }
  wer=$(basename "$checkpoint" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
  awk -v wer="$wer" -v max="$GATE_MAX_WER" 'BEGIN { exit !(wer >= 0 && wer < max) }' || {
    echo "[guided gate] still collapsed after $GATE_EPOCHS epochs: best val_wer=$wer" >&2
    exit 1
  }
  echo "[guided gate] passed: best val_wer=$wer (<$GATE_MAX_WER)"
}

if ! experiment_complete "$NAME" "$FULL_EPOCHS"; then
  completed=$(last_completed_epochs "$NAME")
  if (( completed < GATE_EPOCHS )); then
    echo "=== [guided recovery] resume epoch $completed -> $GATE_EPOCHS ==="
    run_paper_train "$NAME" "$GATE_EPOCHS" "${GUIDED_ARGS[@]}"
  fi
  gate_guided
  run_paper_train "$NAME" "$FULL_EPOCHS" "${GUIDED_ARGS[@]}"
fi

# The safe suite reuses Vanilla/KDBE/Symmetric/Guided when complete, then starts
# S-CTC, FPKD, CARL, CR-CTC and the final Beam-16 evaluation in order.
exec bash experiments/run_chime3_baseline_suite_lr05_safe.sh
