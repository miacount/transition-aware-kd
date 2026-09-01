#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

MASS=data/tedlium2_train.adapted_teacher.d6_mass3_ntdk_m32.json
require_file "$MASS"
export EVAL_MANIFESTS="ted_dev=data/tedlium2_dev.json ted_test=data/tedlium2_test.json"

NAME=paper-ted2-ours-mass25-ntdk5-reliability-s1
run_paper_train "$NAME" 100 \
  --config student_base_ted2 --manifest "$MASS" --kd-mode span_kd \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 5 \
  --span-ntdk-reliability

CKPT=$(best_ckpt "$NAME")
[[ -n "$CKPT" ]] || { echo "missing reliability checkpoint" >&2; exit 1; }
mkdir -p analysis
python scripts/evaluate_ctc_decoding.py \
  --ckpt "$CKPT" --name ted2_ours_mass25_ntdk5_reliability \
  --config configs/student_base_ted2.yaml --beam-width 16 \
  --manifest ted2_dev=data/tedlium2_dev.json \
  --manifest ted2_test=data/tedlium2_test.json \
  | tee analysis/beam16_ted2_ours_mass25_ntdk5_reliability.txt

echo "[done] TED2 Ours reliability training and Beam-16 evaluation"
