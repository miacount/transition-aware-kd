#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SESSION="${DELTA9_TMUX_SESSION:-delta9-ablation}"
LOG=analysis/delta9_ablation_tmux.log
mkdir -p analysis

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && bash experiments/prepare_delta9_targets.sh 2>&1 | tee '$LOG'; status=\${PIPESTATUS[0]}; if [[ \$status -eq 0 ]]; then bash experiments/run_delta9_ablation.sh 2>&1 | tee -a '$LOG'; status=\${PIPESTATUS[0]}; fi; echo DELTA9_EXIT=\$status | tee -a '$LOG'; exec bash"

echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $LOG"
