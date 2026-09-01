#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p analysis
LOG_FILE="${TED2_BASELINE_QUEUE_LOG:-analysis/ted2_new_baselines_queue.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

WAIT_SESSION="${1:-ted2-ours-ws2}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[wait] $(date -u +%FT%TZ) tmux session '$WAIT_SESSION' is still running"
  sleep "$POLL_SECONDS"
done

echo "[start] sweep session '$WAIT_SESSION' finished; launching validated TED2 baseline suite"
exec bash experiments/run_ted2_new_paper_baselines.sh
