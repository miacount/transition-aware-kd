#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/prepare_chime3.py
bash experiments/finetune_teacher_chime3.sh
bash experiments/prepare_chime3_adapted_teacher_targets.sh
bash experiments/run_chime3_students.sh
