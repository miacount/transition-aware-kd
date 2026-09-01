#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

[[ "${FORCE:-0}" != "1" ]] || {
  echo "FORCE=1 is disabled for this staged suite; use new experiment names for a clean rerun" >&2
  exit 2
}

PLAIN=data/chime3_train_enhanced.json
DENSE=data/chime3_train.adapted_teacher.frame_dense_t1.json
SCTC=data/chime3_train.adapted_teacher.sctc.json
CARL=data/chime3_train.adapted_teacher.carl.json
CLASSIFIER=data/carl_chime3_adapted/teacher_classifier.pt
CONFIG=student_base_chime3_full

export EVAL_MANIFESTS="chime3_dev_real=data/chime3_dev_real_enhanced.json chime3_dev_simu=data/chime3_dev_simu_enhanced.json chime3_eval_real=data/chime3_eval_real_enhanced.json chime3_eval_simu=data/chime3_eval_simu_enhanced.json"

for path in "$PLAIN" "$DENSE"; do require_file "$path"; done
bash experiments/prepare_chime3_baseline_targets.sh
for path in "$SCTC" "$CARL" "$CLASSIFIER"; do require_file "$path"; done

python -m pytest -q \
  tests/test_new_paper_losses.py \
  tests/test_average_checkpoints.py \
  tests/test_chime3_runtime_config.py \
  tests/test_chime3_full_config.py \
  tests/test_chime3_baseline_suite.py

# Catch missing/unknown method fields before allocating a long GPU run.
for mode in logit guided sctc fpkd_dfkd fpkd_frkd fpkd_pkd carl_feature carl cr_ctc; do
  python scripts/train.py --cfg job --config-name "$CONFIG" \
    "model.kd_mode=$mode" >/dev/null
done

gate_checkpoint() {
  local name="$1" max_wer="${2:-0.95}" checkpoint wer
  checkpoint=$(best_ckpt "$name")
  [[ -n "$checkpoint" ]] || { echo "[gate] no finite checkpoint: $name" >&2; exit 1; }
  wer=$(basename "$checkpoint" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
  awk -v wer="$wer" -v max="$max_wer" 'BEGIN { exit !(wer >= 0 && wer < max) }' || {
    echo "[gate] collapsed run: $name best val_wer=$wer (required <$max_wer)" >&2
    exit 1
  }
  echo "[gate] $name best val_wer=$wer"
}

run_guarded() {
  local name="$1" full_epochs="$2" smoke_epochs="$3"
  shift 3
  if experiment_complete "$name" "$full_epochs"; then
    echo "=== [reuse] complete experiment: $name ($full_epochs epochs) ==="
    return 0
  fi
  if [[ -z "$(latest_last_ckpt "$name")" ]]; then
    run_paper_train "$name" "$smoke_epochs" "$@"
  fi
  gate_checkpoint "$name"
  run_paper_train "$name" "$full_epochs" "$@"
}

COMMON=(--config "$CONFIG" --lr 0.5 --warmup-steps 750)

run_guarded paper-chime3-vanilla-full-l025-lr05-s1 100 5 \
  "${COMMON[@]}" --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25

run_guarded paper-chime3-kdbe-full-l025-lr05-s1 100 5 \
  "${COMMON[@]}" --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode elimination --kd-lambda 0.25

run_guarded paper-chime3-symmetric-full-l025-n4-lr05-s1 100 5 \
  "${COMMON[@]}" --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric \
  --blank-n 4 --kd-lambda 0.25

run_guarded paper-chime3-guided-exact-w1-lr05-s1 100 5 \
  "${COMMON[@]}" --manifest "$DENSE" --kd-mode guided \
  --temperature 1 --kd-weight 1

SCTC_NAME=paper-chime3-sctc-l1-lr05-s1
run_guarded "$SCTC_NAME" 80 5 \
  "${COMMON[@]}" --manifest "$SCTC" --kd-mode sctc --kd-lambda 1
SCTC_CKPT=$(best_ckpt "$SCTC_NAME")
[[ -n "$SCTC_CKPT" ]] || { echo "missing S-CTC checkpoint" >&2; exit 1; }
run_guarded paper-chime3-sctc-l1-ctcft-lr005-wu100-s1 20 3 \
  --config "$CONFIG" --manifest "$PLAIN" --kd-mode none \
  --init-ckpt "$SCTC_CKPT" --lr 0.05 --warmup-steps 100

DFKD_NAME=paper-chime3-fpkd-stablekl-dfkd-e10-wu300-lr05-s1
FRKD_NAME=paper-chime3-fpkd-stablekl-frkd-e10-wu300-lr05-s1
PKD_NAME=paper-chime3-fpkd-stablekl-pkd-e80-wu2500-lr05-bkl1-nbf1-t1-s1
run_guarded "$DFKD_NAME" 10 3 \
  --config "$CONFIG" --manifest "$DENSE" --kd-mode fpkd_dfkd \
  --fpkd-temp 1 --lr 0.5 --warmup-steps 300
DFKD_CKPT=$(latest_last_ckpt "$DFKD_NAME")
[[ -n "$DFKD_CKPT" ]] || { echo "missing DFKD last checkpoint" >&2; exit 1; }
run_guarded "$FRKD_NAME" 10 3 \
  --config "$CONFIG" --manifest "$CARL" --kd-mode fpkd_frkd \
  --init-ckpt "$DFKD_CKPT" --lr 0.5 --warmup-steps 300
FRKD_CKPT=$(latest_last_ckpt "$FRKD_NAME")
[[ -n "$FRKD_CKPT" ]] || { echo "missing FRKD last checkpoint" >&2; exit 1; }
run_guarded "$PKD_NAME" 80 5 \
  --config "$CONFIG" --manifest "$DENSE" --kd-mode fpkd_pkd \
  --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1 \
  --init-ckpt "$FRKD_CKPT" --lr 0.5 --warmup-steps 2500

CARL_FEATURE=paper-chime3-carl-feature-e10-wu300-lr05-s1
CARL_FULL=paper-chime3-carl-full-ctc-e50-wu1500-lr05-a1-g1-l1-s1
run_guarded "$CARL_FEATURE" 10 3 \
  --config "$CONFIG" --manifest "$CARL" --kd-mode carl_feature \
  --carl-teacher-dim 176 --lr 0.5 --warmup-steps 300
CARL_FEATURE_CKPT=$(latest_last_ckpt "$CARL_FEATURE")
[[ -n "$CARL_FEATURE_CKPT" ]] || { echo "missing CARL feature checkpoint" >&2; exit 1; }
run_guarded "$CARL_FULL" 50 5 \
  --config "$CONFIG" --manifest "$CARL" --kd-mode carl \
  --carl-classifier "$CLASSIFIER" --carl-teacher-dim 176 \
  --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
  --init-ckpt "$CARL_FEATURE_CKPT" --lr 0.5 --warmup-steps 1500 --save-top-k -1
CARL_AVG="nemo_experiments/$CARL_FULL/carl-last10-avg.ckpt"
python scripts/average_checkpoints.py \
  --experiment-dir "nemo_experiments/$CARL_FULL" --last-n 10 --output "$CARL_AVG"

run_guarded paper-chime3-crctc-nemo15-e50-lr05-wu750-s1 50 5 \
  "${COMMON[@]}" --manifest "$PLAIN" --kd-mode cr_ctc \
  --cr-ctc-weight 0.2 --cr-ctc-warm-step 750 --cr-ctc-time-factor 1.5 \
  --train-batch-size 32 --accumulate-grad-batches 2

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_chime3_baseline_suite_lr05.sh
fi

echo "[done] CHiME-3 baseline suite"
