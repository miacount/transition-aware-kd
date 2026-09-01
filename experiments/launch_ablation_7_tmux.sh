#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION="${ABLATION_TMUX_SESSION:-ablation7}"
LOG=analysis/ablation7_tmux.log
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2; exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && bash experiments/prepare_ablation_targets.sh 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; if [[ \$status -eq 0 ]]; then bash experiments/run_ablation_7.sh 2>&1 | tee -a '$LOG'; status=\${PIPESTATUS[0]}; fi; echo ABLATION_EXIT=\$status | tee -a '$LOG'; exec bash"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $LOG"
