#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/chime3_train_enhanced.json
MASS=data/chime3_train.adapted_teacher.d6_mass3_ntdk_m32.json

require_file "$PLAIN"
require_file "$MASS"
export EVAL_MANIFESTS="chime3_dev_real=data/chime3_dev_real_enhanced.json chime3_dev_simu=data/chime3_dev_simu_enhanced.json chime3_eval_real=data/chime3_eval_real_enhanced.json chime3_eval_simu=data/chime3_eval_simu_enhanced.json"

run_paper_train paper-chime3-no-kd-s1 100 \
  --config student_base_chime3_runtime --manifest "$PLAIN" --kd-mode none

run_paper_train paper-chime3-mass3-w25-ntdk8-s1 100 \
  --config student_base_chime3_runtime --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_chime3.sh
fi
