#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

SCOPE="${1:-all}"
case "$SCOPE" in
  lbs|ted3|all) ;;
  *) echo "usage: bash experiments/run_crctc_corrected.sh [lbs|ted3|all]" >&2; exit 2 ;;
esac

# Paper-faithful CR-CTC loss/augmentation with the local baseline's training
# exposure and optimizer schedule. Two views intentionally cost two forwards.
# Physical batch changes are compensated by gradient accumulation so that every
# optimizer update represents 64 utterances, as in the single-view controls.
if [[ "$SCOPE" == "lbs" || "$SCOPE" == "all" ]]; then
  run_paper_train paper-lbs-crctc-corrected-e100-b32a2-mask2p5-s1 100 \
    --config student_base --manifest data/train_clean_100.json --kd-mode cr_ctc \
    --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 \
    --cr-ctc-time-masks-scale 2.5 --cr-ctc-time-width-scale 2.5 \
    --train-batch-size 32 --accumulate-grad-batches 2
fi

if [[ "$SCOPE" == "ted3" || "$SCOPE" == "all" ]]; then
  run_paper_train paper-ted3full-crctc-corrected-e50-b16a4-mask2p5-s1 50 \
    --config student_base_ted3 --manifest data/tedlium3_train.json --kd-mode cr_ctc \
    --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 \
    --cr-ctc-time-masks-scale 2.5 --cr-ctc-time-width-scale 2.5 \
    --train-batch-size 16 --accumulate-grad-batches 4
fi

echo "=== corrected CR-CTC suite complete: $SCOPE ==="
