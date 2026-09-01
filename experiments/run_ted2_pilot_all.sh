#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash experiments/prepare_tedlium2_main.sh

TEACHER=nemo_experiments/teacher-ted2-adapted-s1/best-teacher-ted2-adapted-s1.nemo
if [[ ! -f "$TEACHER" || "${FORCE_TEACHER:-0}" == "1" ]]; then
  echo "=== [teacher] fine-tune LibriSpeech teacher on TED-LIUM2 ==="
  python scripts/finetune_teacher_ted3.py --config-name teacher_finetune_ted2
fi

bash experiments/prepare_ted2_adapted_teacher_targets.sh
RUN_BEAM=0 bash experiments/run_ted2_pilot.sh
bash experiments/evaluate_paper_main.sh ted2

echo "=== TED2 teacher + No-KD + ours pilot complete ==="
