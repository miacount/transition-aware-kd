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

failures=0
run_best() {
  local label="$1" experiment="$2" checkpoint
  checkpoint=$(best_ckpt "$experiment")
  if [[ -z "$checkpoint" ]]; then
    echo "[eval] missing checkpoint: $experiment" >&2
    failures=$((failures + 1))
    return
  fi
  if ! python scripts/evaluate_ctc_decoding.py \
      --ckpt "$checkpoint" --name "$label" \
      --config configs/student_base_chime3_full.yaml \
      --beam-width "$BEAM_WIDTH" "${MANIFESTS[@]}" \
      > "analysis/beam${BEAM_WIDTH}_${label}.txt" 2>&1; then
    echo "[eval] failed: $label (see analysis/beam${BEAM_WIDTH}_${label}.txt)" >&2
    failures=$((failures + 1))
  else
    tail -n 28 "analysis/beam${BEAM_WIDTH}_${label}.txt"
  fi
}

run_exact() {
  local label="$1" checkpoint="$2"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[eval] missing checkpoint: $checkpoint" >&2
    failures=$((failures + 1))
    return
  fi
  if ! python scripts/evaluate_ctc_decoding.py \
      --ckpt "$checkpoint" --name "$label" \
      --config configs/student_base_chime3_full.yaml \
      --beam-width "$BEAM_WIDTH" "${MANIFESTS[@]}" \
      > "analysis/beam${BEAM_WIDTH}_${label}.txt" 2>&1; then
    echo "[eval] failed: $label (see analysis/beam${BEAM_WIDTH}_${label}.txt)" >&2
    failures=$((failures + 1))
  else
    tail -n 28 "analysis/beam${BEAM_WIDTH}_${label}.txt"
  fi
}

run_best chime3_vanilla_l025 paper-chime3-vanilla-full-l025-lr05-s1
run_best chime3_kdbe_l025 paper-chime3-kdbe-full-l025-lr05-s1
run_best chime3_symmetric_l025_n4 paper-chime3-symmetric-full-l025-n4-lr05-s1
run_best chime3_guided_w1 paper-chime3-guided-exact-w1-lr05-s1
run_best chime3_sctc_ctcft paper-chime3-sctc-l1-ctcft-lr005-wu100-s1
run_best chime3_fpkd paper-chime3-fpkd-stablekl-pkd-e80-wu2500-lr05-bkl1-nbf1-t1-s1
run_exact chime3_carl_last10avg \
  nemo_experiments/paper-chime3-carl-full-ctc-e50-wu1500-lr05-a1-g1-l1-s1/carl-last10-avg.ckpt
run_best chime3_crctc paper-chime3-crctc-nemo15-e50-lr05-wu750-s1

if (( failures > 0 )); then
  echo "[eval] completed with $failures failure(s); training checkpoints are intact" >&2
  exit 1
fi
echo "[eval] all CHiME baseline evaluations complete"
