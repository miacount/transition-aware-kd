#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

SOURCE=data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json
BEAM_WIDTH="${BEAM_WIDTH:-16}"
NEW_TOP_M_VALUES="${NEW_TOP_M_VALUES:-1 2 4 16}"
require_file "$SOURCE"
mkdir -p analysis

prepare_topm() {
  local top_m="$1"
  local manifest="data/train_clean_100.teacher_only_d6_mass3_ntdk_m${top_m}.json"
  python scripts/truncate_ntdk_targets.py \
    --manifest-in "$SOURCE" \
    --manifest-out "$manifest" \
    --out-dir "data/span_mass3_ntdk_m${top_m}" \
    --top-m "$top_m" --resume
}

evaluate_topm() {
  local top_m="$1"
  local name="ablation-lbs-ntdk-topm${top_m}-mass25-w8-s1"
  local log="analysis/beam${BEAM_WIDTH}_ablation_lbs_ntdk_topm${top_m}.txt"
  local ckpt
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "no finite checkpoint: $name" >&2; exit 1; }
  if [[ -s "$log" ]] && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" --name "lbs_ntdk_topm${top_m}" \
    --config configs/student_base.yaml --beam-width "$BEAM_WIDTH" \
    --manifest dev_clean=data/dev_clean.json \
    --manifest dev_other=data/dev_other.json \
    --manifest test_clean=data/test_clean.json \
    --manifest test_other=data/test_other.json | tee "$log"
}

for top_m in $NEW_TOP_M_VALUES; do
  case "$top_m" in
    1|2|4|16) ;;
    *) echo "NEW_TOP_M_VALUES must contain only 1, 2, 4, or 16" >&2; exit 2 ;;
  esac
  manifest="data/train_clean_100.teacher_only_d6_mass3_ntdk_m${top_m}.json"
  prepare_topm "$top_m"
  run_paper_train "ablation-lbs-ntdk-topm${top_m}-mass25-w8-s1" 100 \
    --config student_base --manifest "$manifest" --kd-mode span_kd \
    --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8
  evaluate_topm "$top_m"
done

# Existing M=3/8 ablations and the frozen M=32 paper-main result are deliberately
# not retrained. The summarizer consumes their beam-16 logs directly.
for top_m in 3 8; do
  require_file "analysis/beam${BEAM_WIDTH}_ablation_lbs_ntdk_topm${top_m}.txt"
done
experiment_complete span-d6-mass3-w25-ntdk-a8-s1 100 || {
  echo "existing M=32 experiment is incomplete" >&2; exit 1;
}
require_file "analysis/beam${BEAM_WIDTH}_paper_lbs_mass25_ntdk8.txt"
python scripts/summarize_ntdk_topm_ablation.py --beam-width "$BEAM_WIDTH"

echo "=== NTDK top-M ablation complete (M=1,2,4,16 added; M=3,8,32 reused) ==="
