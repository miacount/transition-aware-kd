#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/tedlium2_train.json
MASS=data/tedlium2_train.adapted_teacher.d6_mass3_ntdk_m32.json

require_file "$PLAIN"
require_file "$MASS"
export EVAL_MANIFESTS="ted_dev=data/tedlium2_dev.json ted_test=data/tedlium2_test.json"

run_paper_train paper-ted2-no-kd-s1 100 \
  --config student_base_ted2 --manifest "$PLAIN" --kd-mode none

# Same delta/weights/top-M as LibriSpeech: this is a fixed transfer, not TED tuning.
run_paper_train paper-ted2-mass3-w25-ntdk8-s1 100 \
  --config student_base_ted2 --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  bash experiments/evaluate_paper_main.sh ted2
fi
