#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash experiments/prepare_ted3_adapted_teacher_targets.sh
RUN_BEAM=0 bash experiments/run_paper_main_ted3_adapted_teacher.sh
bash experiments/evaluate_paper_main.sh ted3adapt

echo "=== TED3 adapted-teacher reproduction suite complete ==="
