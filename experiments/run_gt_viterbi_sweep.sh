#!/usr/bin/env bash
# Re-run all experiments affected by GT-Viterbi alignment change.
# Run sequentially (single GPU). Takes ~8–9 hours total.
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.token_avg.json
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST; run 30_build_token_avg_targets.sh first" >&2; exit 1; }

echo "=========================================="
echo "1/5  token_avg w=20  (GT-Viterbi v3)"
echo "=========================================="
bash experiments/train.sh \
  --name student-token-avg-gt-w20-clean \
  --manifest "$MANIFEST" \
  --kd-mode token_avg \
  --kd-weight 20.0

echo "=========================================="
echo "2/5  aligned_token w=20  (GT-Viterbi)"
echo "=========================================="
bash experiments/train.sh \
  --name student-aligned-token-gt-w20-clean \
  --manifest "$MANIFEST" \
  --kd-mode aligned_token \
  --kd-weight 20.0

echo "=========================================="
echo "3/5  token_avg w=20 + transition w=5"
echo "=========================================="
bash experiments/train.sh \
  --name student-token-avg-gt-w20-tw5-clean \
  --manifest "$MANIFEST" \
  --kd-mode token_avg \
  --kd-weight 20.0 \
  --trans-kd-weight 5.0

echo "=========================================="
echo "4/5  token_avg w=20 + transition w=10"
echo "=========================================="
bash experiments/train.sh \
  --name student-token-avg-gt-w20-tw10-clean \
  --manifest "$MANIFEST" \
  --kd-mode token_avg \
  --kd-weight 20.0 \
  --trans-kd-weight 10.0

echo "=========================================="
echo "5/5  token_avg w=20 + transition w=20"
echo "=========================================="
bash experiments/train.sh \
  --name student-token-avg-gt-w20-tw20-clean \
  --manifest "$MANIFEST" \
  --kd-mode token_avg \
  --kd-weight 20.0 \
  --trans-kd-weight 20.0

echo "=========================================="
echo "All done."
echo "=========================================="
