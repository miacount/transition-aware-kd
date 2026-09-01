#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

BEAM_WIDTH="${BEAM_WIDTH:-16}"
mkdir -p analysis
MANIFESTS=(
  --manifest chime3_dev_real=data/chime3_dev_real_enhanced.json
  --manifest chime3_dev_simu=data/chime3_dev_simu_enhanced.json
  --manifest chime3_eval_real=data/chime3_eval_real_enhanced.json
  --manifest chime3_eval_simu=data/chime3_eval_simu_enhanced.json
)

for spec in \
  "chime3_lr05_nokd paper-chime3-lr05-no-kd-s1" \
  "chime3_lr05_mass25_ntdk8 paper-chime3-lr05-mass3-w25-ntdk8-s1"; do
  read -r label experiment <<< "$spec"
  checkpoint=$(best_ckpt "$experiment")
  [[ -n "$checkpoint" ]] || { echo "missing checkpoint for $experiment" >&2; exit 1; }
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$checkpoint" --name "$label" \
    --config configs/student_base_chime3_runtime.yaml \
    --beam-width "$BEAM_WIDTH" "${MANIFESTS[@]}" \
    | tee "analysis/beam${BEAM_WIDTH}_paper_${label}.txt"
done
