#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash experiments/prepare_tedlium3_main.sh
bash experiments/prepare_paper_targets.sh all
RUN_BEAM=0 bash experiments/run_paper_main_lbs.sh
RUN_BEAM=0 bash experiments/run_paper_main_ted3.sh
bash experiments/evaluate_paper_main.sh lbs
bash experiments/evaluate_paper_main.sh ted3
python scripts/summarize_paper_main.py --beam-width "${BEAM_WIDTH:-16}"

echo "=== LibriSpeech-100 + TED-LIUM3-full main-table suite complete ==="
