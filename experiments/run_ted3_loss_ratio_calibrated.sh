#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

DENSE=data/tedlium3_train.adapted_teacher.frame_dense_t1.json
MASS=data/tedlium3_train.adapted_teacher.d6_mass3_ntdk_m32.json

require_file "$DENSE"
require_file "$MASS"
export EVAL_MANIFESTS="ted3_dev=data/tedlium3_dev.json ted3_test=data/tedlium3_test.json"

# Preserve each LibriSpeech run's median weighted auxiliary/CTC ratio over its
# final five epochs. See analysis/ted3_loss_ratio_calibration_2026-08-05.md.

run_paper_train paper-ted3adapt-ratiocal-vanilla-l0289-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode none \
  --kd-lambda 0.288849

run_paper_train paper-ted3adapt-ratiocal-kdbe-l0439-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode elimination \
  --kd-lambda 0.439375

run_paper_train paper-ted3adapt-ratiocal-symmetric-l0310-n2-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric \
  --blank-n 2 --kd-lambda 0.310436

run_paper_train paper-ted3adapt-ratiocal-guided-w1202-s1 50 \
  --config student_base_ted3 --manifest "$DENSE" --kd-mode guided \
  --temperature 1 --kd-weight 1.202025

run_paper_train paper-ted3adapt-ratiocal-mass3-w16p53-ntdk4p85-s1 50 \
  --config student_base_ted3 --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 16.532378 --span-primary-mode mass3 \
  --span-ntdk-weight 4.853598

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_paper_main.sh ted3ratiocal
fi
