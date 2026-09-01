#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/tedlium3_train.json
DENSE=data/tedlium3_train.adapted_teacher.frame_dense_t1.json
MASS=data/tedlium3_train.adapted_teacher.d6_mass3_ntdk_m32.json
SCTC=data/tedlium3_train.adapted_teacher.sctc.json

require_file "$PLAIN"
require_file "$DENSE"
require_file "$MASS"
require_file "$SCTC"
export EVAL_MANIFESTS="ted3_dev=data/tedlium3_dev.json ted3_test=data/tedlium3_test.json"

# No-KD and CR-CTC do not consume a teacher, so their existing TED3 runs remain valid.
run_paper_train paper-ted3adapt-vanilla-full-l09-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.9

run_paper_train paper-ted3adapt-kdbe-full-l09-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode elimination --kd-lambda 0.9

run_paper_train paper-ted3adapt-symmetric-full-l1-n2-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric \
  --blank-n 2 --kd-lambda 1

run_paper_train paper-ted3adapt-guided-exact-w1-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode guided \
  --temperature 1 --kd-weight 1

run_paper_train paper-ted3adapt-sctc-l1-s1 40 \
  --config student_base_ted3 --manifest "$SCTC" --kd-mode sctc --kd-lambda 1
SCTC_CKPT=$(best_ckpt paper-ted3adapt-sctc-l1-s1)
[[ -n "$SCTC_CKPT" ]] || { echo "no adapted-teacher S-CTC checkpoint found" >&2; exit 1; }
run_paper_train paper-ted3adapt-sctc-l1-ctcft-s1 10 \
  --config student_base_ted3 --manifest "$PLAIN" --kd-mode none \
  --init-ckpt "$SCTC_CKPT"

run_paper_train paper-ted3adapt-mass3-w25-ntdk8-s1 50 \
  --config student_base_ted3 --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_paper_main.sh ted3adapt
fi
