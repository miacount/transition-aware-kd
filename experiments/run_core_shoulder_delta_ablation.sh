#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

SHOULDER_WEIGHT="${SHOULDER_WEIGHT:-1}"
name="ablation-lbs-core0-shoulder-d6-w${SHOULDER_WEIGHT}-s1"
manifest="data/train_clean_100.core0_shoulder_d6_mass3_ntdk_m32.json"
require_file "$manifest"
run_paper_train "$name" 100 \
  --config student_base --manifest "$manifest" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8 \
  --span-shoulder-weight "$SHOULDER_WEIGHT"
