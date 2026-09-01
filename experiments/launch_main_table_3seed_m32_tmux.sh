#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SESSION="${MAIN3_TMUX_SESSION:-main3-m32}"
LOG=analysis/main_table_3seed_m32_tmux.log
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && bash experiments/run_main_table_3seed_m32.sh 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; echo MAIN3_M32_EXIT=\$status | tee -a '$LOG'; exec bash"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $LOG"
