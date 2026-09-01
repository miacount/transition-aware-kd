#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash experiments/run_chime3_baseline_suite_lr05_safe.sh "$@"
