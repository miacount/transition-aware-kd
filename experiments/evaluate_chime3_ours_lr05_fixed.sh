#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

NAME=paper-chime3-lr05-mass3-w25-ntdk8-s1
BEAM_WIDTH="${BEAM_WIDTH:-16}"
CKPT=$(best_ckpt "$NAME")
[[ -n "$CKPT" ]] || { echo "missing checkpoint for $NAME" >&2; exit 1; }

mkdir -p analysis
python scripts/evaluate_ctc_decoding.py \
  --ckpt "$CKPT" --name chime3_lr05_mass25_ntdk8 \
  --config configs/student_base_ted3.yaml \
  --beam-width "$BEAM_WIDTH" \
  --manifest chime3_dev_real=data/chime3_dev_real_enhanced.json \
  --manifest chime3_dev_simu=data/chime3_dev_simu_enhanced.json \
  --manifest chime3_eval_real=data/chime3_eval_real_enhanced.json \
  --manifest chime3_eval_simu=data/chime3_eval_simu_enhanced.json \
  | tee "analysis/beam${BEAM_WIDTH}_paper_chime3_lr05_mass25_ntdk8.txt"
