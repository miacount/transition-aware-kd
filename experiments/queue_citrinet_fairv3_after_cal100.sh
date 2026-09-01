#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CURRENT_PID="${1:-}"

if [[ -n "$CURRENT_PID" ]]; then
  echo "[$(date -u +'%F %T UTC')] waiting for calibrated suite pid=$CURRENT_PID"
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 60
  done
fi

echo "[$(date -u +'%F %T UTC')] starting equal-budget dev-only tuning"
bash experiments/run_citrinet144_equal_budget_tuning.sh \
  2>&1 | tee analysis/citrinet144_tune30_run.log

echo "[$(date -u +'%F %T UTC')] starting frozen fair-v3 seed-1 suite"
bash experiments/run_citrinet144_fairv3_selected.sh \
  2>&1 | tee analysis/citrinet144_fairv3_run.log

echo "[$(date -u +'%F %T UTC')] fair-v3 queue complete"
