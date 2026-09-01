#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/tedlium3_train.json
DENSE=data/tedlium3_train.small_teacher.frame_dense_t1.json
MASS=data/tedlium3_train.teacher_only_d6_mass3_ntdk_m32.json
SCTC=data/tedlium3_train.sctc.json

require_file "$PLAIN"
require_file "$DENSE"
require_file "$MASS"
require_file "$SCTC"
export EVAL_MANIFESTS="ted3_dev=data/tedlium3_dev.json ted3_test=data/tedlium3_test.json"

# All single-view methods receive 50 full TED-LIUM3 epochs.
run_paper_train paper-ted3full-no-kd-s1 50 \
  --config student_base_ted3 --manifest "$PLAIN" --kd-mode none

# Hilmes et al. hyperparameters are fixed a priori from their TED-LIUM2 table.
run_paper_train paper-ted3full-vanilla-full-l09-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.9

run_paper_train paper-ted3full-kdbe-full-l09-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode elimination --kd-lambda 0.9

run_paper_train paper-ted3full-symmetric-full-l1-n2-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric \
  --blank-n 2 --kd-lambda 1

# Exact Guided CTC: L_CTC - sum of selected student posteriors.
run_paper_train paper-ted3full-guided-exact-w1-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode guided \
  --temperature 1 --kd-weight 1

# S-CTC needs its defining CTC fine-tuning stage. Keep total exposure at 50
# epochs by allocating 40 to S-CTC and 10 to supervised CTC fine-tuning.
run_paper_train paper-ted3full-sctc-l1-s1 40 \
  --config student_base_ted3 --manifest "$SCTC" --kd-mode sctc --kd-lambda 1
SCTC_CKPT=$(best_ckpt paper-ted3full-sctc-l1-s1)
[[ -n "$SCTC_CKPT" ]] || { echo "no TED3 S-CTC checkpoint found" >&2; exit 1; }
run_paper_train paper-ted3full-sctc-l1-ctcft-s1 10 \
  --config student_base_ted3 --manifest "$PLAIN" --kd-mode none \
  --init-ckpt "$SCTC_CKPT"

# CR-CTC is compute-matched: two views, half the epochs and half physical
# batch, preserving the single-view suite's optimizer-step/forward budget.
run_paper_train paper-ted3full-crctc-fair25-b16-mask2p5-s1 25 \
  --config student_base_ted3 --manifest "$PLAIN" --kd-mode cr_ctc \
  --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 \
  --cr-ctc-time-masks-scale 2.5 --cr-ctc-time-width-scale 2.5 \
  --train-batch-size 16 --accumulate-grad-batches 2

# Proposed method is transferred unchanged from LibriSpeech; no TED tuning.
run_paper_train paper-ted3full-mass3-w25-ntdk8-s1 50 \
  --config student_base_ted3 --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_paper_main.sh ted3
fi
