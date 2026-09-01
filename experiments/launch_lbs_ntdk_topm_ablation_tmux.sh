#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SESSION="${NTDK_TOPM_TMUX_SESSION:-ntdk-topm-extended}"
LOG="analysis/ntdk_topm_extended_ablation_tmux.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && bash experiments/run_lbs_ntdk_topm_ablation.sh 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; echo NTDK_TOPM_EXIT=\$status | tee -a '$LOG'; exec bash"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $LOG"
