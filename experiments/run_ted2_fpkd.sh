#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

DENSE=data/tedlium2_train.adapted_teacher.frame_dense_t1.json
CARL=data/tedlium2_train.adapted_teacher.carl.json
CLASSIFIER=data/carl_ted2_adapted/teacher_classifier.pt
require_file "$DENSE"
if [[ ! -f "$CARL" || ! -f "$CLASSIFIER" ]]; then
  bash experiments/prepare_ted2_carl_targets.sh
fi
require_file "$CARL"
require_file "$CLASSIFIER"
export EVAL_MANIFESTS="ted_dev=data/tedlium2_dev.json ted_test=data/tedlium2_test.json"

# Exact 0/1 posterior targets must remain finite before a long progressive run.
python -m pytest -q tests/test_new_paper_losses.py tests/test_new_paper_protocol.py

DFKD_EPOCHS="${FPKD_DFKD_EPOCHS:-10}"
FRKD_EPOCHS="${FPKD_FRKD_EPOCHS:-10}"
PKD_EPOCHS="${FPKD_PKD_EPOCHS:-80}"
DFKD_WARMUP="${FPKD_DFKD_WARMUP:-1000}"
FRKD_WARMUP="${FPKD_FRKD_WARMUP:-1000}"
PKD_WARMUP="${FPKD_PKD_WARMUP:-8000}"

DFKD_NAME=paper-ted2-fpkd-stablekl-dfkd-e${DFKD_EPOCHS}-wu${DFKD_WARMUP}-s1
FRKD_NAME=paper-ted2-fpkd-stablekl-frkd-e${FRKD_EPOCHS}-wu${FRKD_WARMUP}-s1
PKD_NAME=paper-ted2-fpkd-stablekl-pkd-e${PKD_EPOCHS}-wu${PKD_WARMUP}-bkl1-nbf1-t1-s1

# The stablekl names are intentionally distinct from the stopped NaN run.
# A later interruption of this corrected run can still resume safely.
run_paper_train "$DFKD_NAME" "$DFKD_EPOCHS" \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode fpkd_dfkd \
  --fpkd-temp 1 --warmup-steps "$DFKD_WARMUP"
DFKD_CKPT=$(latest_last_ckpt "$DFKD_NAME")
[[ -n "$DFKD_CKPT" ]] || { echo "missing DFKD last checkpoint" >&2; exit 1; }

run_paper_train "$FRKD_NAME" "$FRKD_EPOCHS" \
  --config student_base_ted2 --manifest "$CARL" --kd-mode fpkd_frkd \
  --warmup-steps "$FRKD_WARMUP" --init-ckpt "$DFKD_CKPT"
FRKD_CKPT=$(latest_last_ckpt "$FRKD_NAME")
[[ -n "$FRKD_CKPT" ]] || { echo "missing FRKD last checkpoint" >&2; exit 1; }

run_paper_train "$PKD_NAME" "$PKD_EPOCHS" \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode fpkd_pkd \
  --fpkd-bkl-weight 1 --fpkd-nbf-weight 1 --fpkd-temp 1 \
  --warmup-steps "$PKD_WARMUP" --init-ckpt "$FRKD_CKPT"
