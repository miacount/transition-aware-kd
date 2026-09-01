#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

MANIFEST=data/tedlium2_train.adapted_teacher.d6_mass3_ntdk_m32.json
require_file "$MANIFEST"
export EVAL_MANIFESTS="ted_dev=data/tedlium2_dev.json ted_test=data/tedlium2_test.json"

# Two highest-probability settings from TED2/LS loss-scale analysis:
# 22/6.5 = LS loss-ratio transfer; 25/5 = retain the strong Mass3 anchor while
# reducing the TED2-overstrong conditional NTDK branch.
run_paper_train paper-ted2-ours-mass22-ntdk6p5-s1 100 \
  --config student_base_ted2 --manifest "$MANIFEST" --kd-mode span_kd \
  --kd-weight 22 --span-primary-mode mass3 --span-ntdk-weight 6.5
run_paper_train paper-ted2-ours-mass25-ntdk5-s1 100 \
  --config student_base_ted2 --manifest "$MANIFEST" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 5
