#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TEACHER_OUT="nemo_experiments/teacher-ted3-adapted-s1/best-teacher-ted3-adapted-s1.nemo"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "=== waiting for TED3 teacher fine-tuning to finish ==="
while pgrep -f '^python scripts/finetune_teacher_ted3.py' >/dev/null; do
  sleep "$POLL_SECONDS"
done

if [[ ! -s "$TEACHER_OUT" ]]; then
  echo "teacher fine-tuning ended without the expected export: $TEACHER_OUT" >&2
  echo "CR-CTC was not started; inspect the teacher run logs first." >&2
  exit 1
fi

echo "=== teacher export found: $TEACHER_OUT ==="
echo "=== starting NeMo-adapted CR-CTC (LBS then TED3) ==="
bash experiments/run_crctc_nemo_adapted.sh all
