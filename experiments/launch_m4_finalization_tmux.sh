#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SESSION="${M4_FINAL_TMUX_SESSION:-m4-finalization}"
LOG=analysis/m4_finalization_tmux.log
mkdir -p analysis

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && bash experiments/prepare_m4_finalization_targets.sh 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; if [[ \$status -eq 0 ]]; then bash experiments/run_m4_finalization_suite.sh 2>&1 | tee -a '$LOG'; status=\${PIPESTATUS[0]}; fi; echo M4_FINAL_EXIT=\$status | tee -a '$LOG'; exec bash"

echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $LOG"
