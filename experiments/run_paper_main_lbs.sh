#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

DENSE=data/train_clean_100.small_teacher.frame_dense_t1.json
MASS=data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json
SCTC=data/train_clean_100.sctc.json
PLAIN=data/train_clean_100.json

require_file "$PLAIN"
require_file "$DENSE"
require_file "$MASS"
require_file "$SCTC"
unset EVAL_MANIFESTS || true

# Common no-KD control (already complete in the current workspace).
run_paper_train student-no-kd 100 \
  --config student_base --manifest "$PLAIN" --kd-mode none

# Hilmes et al. full-posterior settings selected a priori from their LBS table.
run_paper_train paper-lbs-vanilla-full-l025-s1 100 \
  --config student_base --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25

run_paper_train paper-lbs-kdbe-full-l025-s1 100 \
  --config student_base --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode elimination --kd-lambda 0.25

run_paper_train paper-lbs-symmetric-full-l025-n4-s1 100 \
  --config student_base --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric --blank-n 4 --kd-lambda 0.25

# Kurata & Audhkhasi: L_CTC + L_G, with L_G=-sum selected student posterior.
run_paper_train paper-lbs-guided-exact-w1-s1 100 \
  --config student_base --manifest "$DENSE" --kd-mode guided \
  --temperature 1 --kd-weight 1

# S-CTC needs its CTC fine-tuning stage. Keep total exposure at 100 epochs.
run_paper_train paper-lbs-sctc-l1-s1 80 \
  --config student_base --manifest "$SCTC" --kd-mode sctc --kd-lambda 1
SCTC_CKPT=$(best_ckpt paper-lbs-sctc-l1-s1)
[[ -n "$SCTC_CKPT" ]] || { echo "no S-CTC checkpoint found" >&2; exit 1; }
run_paper_train paper-lbs-sctc-l1-ctcft-s1 20 \
  --config student_base --manifest "$PLAIN" --kd-mode none --init-ckpt "$SCTC_CKPT"

# Stable compute-matched CR-CTC run used in the table: 2 views and half epochs.
# This is the validated local recipe (legacy sqrt-scaled mask factor 1.5); the
# stricter 2.5x-count/2.5x-width reproduction collapsed and is kept diagnostic.
run_paper_train cr_ctc_w02_fair50 50 \
  --config student_base --manifest "$PLAIN" --kd-mode cr_ctc \
  --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 \
  --cr-ctc-time-factor 1.5

# Proposed method: frozen before cross-domain evaluation; no further tuning here.
run_paper_train span-d6-mass3-w25-ntdk-a8-s1 100 \
  --config student_base --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_paper_main.sh lbs
fi
