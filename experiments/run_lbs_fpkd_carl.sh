#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

DENSE=data/train_clean_100.small_teacher.frame_dense_t1.json
CARL=data/train_clean_100.small_teacher.carl.json
CLASSIFIER=data/carl_lbs_small/teacher_classifier.pt
require_file "$DENSE"
if [[ ! -f "$CARL" || ! -f "$CLASSIFIER" ]]; then
  bash experiments/prepare_lbs_carl_targets.sh
fi
require_file "$CARL"
require_file "$CLASSIFIER"
unset EVAL_MANIFESTS || true

python -m pytest -q tests/test_new_paper_losses.py \
  tests/test_average_checkpoints.py tests/test_new_paper_protocol.py

DFKD_EPOCHS="${FPKD_DFKD_EPOCHS:-10}"
FRKD_EPOCHS="${FPKD_FRKD_EPOCHS:-10}"
PKD_EPOCHS="${FPKD_PKD_EPOCHS:-80}"
DFKD_WARMUP="${FPKD_DFKD_WARMUP:-1000}"
FRKD_WARMUP="${FPKD_FRKD_WARMUP:-1000}"
PKD_WARMUP="${FPKD_PKD_WARMUP:-8000}"
DFKD_NAME=paper-lbs-fpkd-stablekl-dfkd-e${DFKD_EPOCHS}-wu${DFKD_WARMUP}-s1
FRKD_NAME=paper-lbs-fpkd-stablekl-frkd-e${FRKD_EPOCHS}-wu${FRKD_WARMUP}-s1
PKD_NAME=paper-lbs-fpkd-stablekl-pkd-e${PKD_EPOCHS}-wu${PKD_WARMUP}-bkl1-nbf1-t1-s1

run_paper_train "$DFKD_NAME" "$DFKD_EPOCHS" \
  --config student_base --manifest "$DENSE" --kd-mode fpkd_dfkd \
  --fpkd-temp 1 --warmup-steps "$DFKD_WARMUP"
DFKD_CKPT=$(latest_last_ckpt "$DFKD_NAME")
[[ -n "$DFKD_CKPT" ]] || { echo "missing LS DFKD checkpoint" >&2; exit 1; }

run_paper_train "$FRKD_NAME" "$FRKD_EPOCHS" \
  --config student_base --manifest "$CARL" --kd-mode fpkd_frkd \
  --warmup-steps "$FRKD_WARMUP" --init-ckpt "$DFKD_CKPT"
FRKD_CKPT=$(latest_last_ckpt "$FRKD_NAME")
[[ -n "$FRKD_CKPT" ]] || { echo "missing LS FRKD checkpoint" >&2; exit 1; }

run_paper_train "$PKD_NAME" "$PKD_EPOCHS" \
  --config student_base --manifest "$DENSE" --kd-mode fpkd_pkd \
  --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1 \
  --warmup-steps "$PKD_WARMUP" --init-ckpt "$FRKD_CKPT"
FPKD_CKPT=$(best_ckpt "$PKD_NAME")
[[ -n "$FPKD_CKPT" ]] || { echo "missing LS PKD best checkpoint" >&2; exit 1; }
python scripts/evaluate_ctc_decoding.py \
  --ckpt "$FPKD_CKPT" --name lbs_fpkd_stablekl \
  --config configs/student_base.yaml --beam-width 16 \
  --manifest dev_clean=data/dev_clean.json --manifest dev_other=data/dev_other.json \
  --manifest test_clean=data/test_clean.json --manifest test_other=data/test_other.json \
  | tee analysis/beam16_lbs_fpkd_stablekl.txt

FEATURE_NAME=paper-lbs-carl-feature-e10-wu1000-s1
FULL_NAME=paper-lbs-carl-full-ctc-e50-wu5000-a1-g1-l1-s1
run_paper_train "$FEATURE_NAME" 10 \
  --config student_base --manifest "$CARL" --kd-mode carl_feature \
  --carl-teacher-dim 176 --warmup-steps 1000
FEATURE_CKPT=$(latest_last_ckpt "$FEATURE_NAME")
[[ -n "$FEATURE_CKPT" ]] || { echo "missing LS CARL feature checkpoint" >&2; exit 1; }

run_paper_train "$FULL_NAME" 50 \
  --config student_base --manifest "$CARL" --kd-mode carl \
  --carl-classifier "$CLASSIFIER" --carl-teacher-dim 176 \
  --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
  --warmup-steps 5000 --save-top-k -1 --init-ckpt "$FEATURE_CKPT"

CARL_DIR=nemo_experiments/$FULL_NAME
CARL_AVG=$CARL_DIR/carl-last10-avg.ckpt
python scripts/average_checkpoints.py \
  --experiment-dir "$CARL_DIR" --last-n 10 --output "$CARL_AVG"
python scripts/evaluate_ctc_decoding.py \
  --ckpt "$CARL_AVG" --name lbs_carl_full_last10avg \
  --config configs/student_base.yaml --beam-width 16 \
  --manifest dev_clean=data/dev_clean.json --manifest dev_other=data/dev_other.json \
  --manifest test_clean=data/test_clean.json --manifest test_other=data/test_other.json \
  | tee analysis/beam16_lbs_carl_full_last10avg.txt

echo "[done] LS FPKD and Full CARL training/evaluation"
