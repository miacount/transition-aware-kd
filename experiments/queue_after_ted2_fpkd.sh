#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p analysis
WAIT_SESSION="${1:-ted2-fpkd-fixed}"
POLL_SECONDS="${POLL_SECONDS:-30}"

echo "[queue] waiting for tmux session: $WAIT_SESSION"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[wait] $(date -u +%FT%TZ) $WAIT_SESSION is still running"
  sleep "$POLL_SECONDS"
done

echo "[1/3] $(date -u +%FT%TZ) TED2 Ours reliability"
bash experiments/run_ted2_ours_reliability.sh

echo "[2/3] $(date -u +%FT%TZ) LS cache preparation and FPKD"
echo "[3/3] LS Full CARL follows FPKD inside the same validated script"
bash experiments/run_lbs_fpkd_carl.sh

echo "[queue done] $(date -u +%FT%TZ)"
