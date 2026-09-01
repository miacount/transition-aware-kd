#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

SCOPE="${1:-all}"
case "$SCOPE" in
  lbs|ted3|all) ;;
  *) echo "usage: bash experiments/run_crctc_nemo_adapted.sh [lbs|ted3|all]" >&2; exit 2 ;;
esac

# CR-CTC objective: official dual-view mean CTC + stop-gradient symmetric KL,
# alpha=0.2, 2000-update consistency warm-up.
#
# SpecAugment is framework-adapted. The literal 2.5x count AND 2.5x width port
# collapses this NeMo Conformer (36.27% LBS dev WER after 100 epochs). A volume
# factor of 1.5 maps to sqrt(1.5) per mask dimension in src/model.py and is the
# strongest local setting already verified to converge (12.16% LBS dev WER).
# Full dataset exposure is retained. Batch/accumulation always represents 64
# utterances per optimizer update, preserving the baseline Noam schedule.

if [[ "$SCOPE" == "lbs" || "$SCOPE" == "all" ]]; then
  run_paper_train paper-lbs-crctc-nemo15-e100-b32a2-s1 100 \
    --config student_base --manifest data/train_clean_100.json --kd-mode cr_ctc \
    --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 \
    --cr-ctc-time-factor 1.5 \
    --train-batch-size 32 --accumulate-grad-batches 2
fi

if [[ "$SCOPE" == "ted3" || "$SCOPE" == "all" ]]; then
  run_paper_train paper-ted3full-crctc-nemo15-e50-b16a4-s1 50 \
    --config student_base_ted3 --manifest data/tedlium3_train.json --kd-mode cr_ctc \
    --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 \
    --cr-ctc-time-factor 1.5 \
    --train-batch-size 16 --accumulate-grad-batches 4
fi

echo "=== NeMo-adapted CR-CTC suite complete: $SCOPE ==="
