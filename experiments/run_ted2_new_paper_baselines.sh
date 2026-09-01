#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/tedlium2_train.json
DENSE=data/tedlium2_train.adapted_teacher.frame_dense_t1.json
CARL=data/tedlium2_train.adapted_teacher.carl.json
CLASSIFIER=data/carl_ted2_adapted/teacher_classifier.pt
require_file "$PLAIN"
require_file "$DENSE"
if [[ ! -f "$CARL" || ! -f "$CLASSIFIER" ]]; then
  bash experiments/prepare_ted2_carl_targets.sh
fi
require_file "$CARL"
require_file "$CLASSIFIER"
export EVAL_MANIFESTS="ted_dev=data/tedlium2_dev.json ted_test=data/tedlium2_test.json"

# Refuse to spend a multi-day GPU run on an implementation that has not passed
# the loss and checkpoint-protocol regression tests.
python -m pytest -q tests/test_new_paper_losses.py tests/test_average_checkpoints.py \
  tests/test_new_paper_protocol.py

# Frame-DKD adaptation: original DKD alpha/beta/T/warmup; teacher top-1 is the
# explicitly declared CTC frame pseudo-target. Temperature is reconstructed
# from cached T=1 probabilities without a log floor. Total compute matches no-KD.
run_paper_train paper-ted2-frame-dkd-teacherargmax-a1-b8-t4-w20-s1 100 \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode frame_dkd \
  --frame-dkd-alpha 1 --frame-dkd-beta 8 --frame-dkd-temp 4 --frame-dkd-warmup 20

# FPKD CTC adaptation. The public article confirms the three objectives but
# does not expose its exact stage schedule and all coefficients. Keep those
# assumptions explicit in experiment names and paper notes.
DFKD_EPOCHS="${FPKD_DFKD_EPOCHS:-10}"
FRKD_EPOCHS="${FPKD_FRKD_EPOCHS:-10}"
PKD_EPOCHS="${FPKD_PKD_EPOCHS:-80}"
DFKD_WARMUP="${FPKD_DFKD_WARMUP:-1000}"
FRKD_WARMUP="${FPKD_FRKD_WARMUP:-1000}"
PKD_WARMUP="${FPKD_PKD_WARMUP:-8000}"
DFKD_NAME=paper-ted2-fpkd-stablekl-dfkd-e${DFKD_EPOCHS}-wu${DFKD_WARMUP}-s1
FRKD_NAME=paper-ted2-fpkd-stablekl-frkd-e${FRKD_EPOCHS}-wu${FRKD_WARMUP}-s1
PKD_NAME=paper-ted2-fpkd-stablekl-pkd-e${PKD_EPOCHS}-wu${PKD_WARMUP}-bkl1-nbf1-t1-s1
# stablekl names deliberately prevent reuse of the stopped NaN run.
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

# Full CTC-side CARL: the paper's TED2 schedule is feature KD 10 epochs then
# CARL 50 epochs. Stage 2 includes both heads' CTC/BE terms; only the source
# paper's attention-decoder CE is inapplicable to this CTC-only architecture.
run_paper_train paper-ted2-carl-feature-e10-wu1000-s1 10 \
  --config student_base_ted2 --manifest "$CARL" --kd-mode carl_feature \
  --carl-teacher-dim 176 --warmup-steps 1000
CARL_FEATURE_CKPT=$(latest_last_ckpt paper-ted2-carl-feature-e10-wu1000-s1)
[[ -n "$CARL_FEATURE_CKPT" ]] || { echo "missing CARL feature last checkpoint" >&2; exit 1; }
run_paper_train paper-ted2-carl-full-ctc-e50-wu5000-a1-g1-l1-s1 50 \
  --config student_base_ted2 --manifest "$CARL" --kd-mode carl \
  --carl-classifier "$CLASSIFIER" --carl-teacher-dim 176 \
  --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
  --warmup-steps 5000 --save-top-k -1 --init-ckpt "$CARL_FEATURE_CKPT"

# Tian et al. report averaging the final ten epochs. Preserve and evaluate that
# checkpoint separately from the shared dev-best checkpoint evaluation.
CARL_DIR=nemo_experiments/paper-ted2-carl-full-ctc-e50-wu5000-a1-g1-l1-s1
CARL_AVG="$CARL_DIR/carl-last10-avg.ckpt"
python scripts/average_checkpoints.py \
  --experiment-dir "$CARL_DIR" --last-n 10 --output "$CARL_AVG"
bash experiments/eval.sh --ckpt "$CARL_AVG" \
  --name paper-ted2-carl-full-ctc-last10avg-a1-g1-l1-s1 \
  --config configs/student_base_ted2.yaml
