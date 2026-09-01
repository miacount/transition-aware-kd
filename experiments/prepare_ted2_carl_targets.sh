#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER="${TED2_ADAPTED_TEACHER:-nemo_experiments/teacher-ted2-adapted-s1/best-teacher-ted2-adapted-s1.nemo}"
DENSE=data/tedlium2_train.adapted_teacher.frame_dense_t1.json
OUT=data/tedlium2_train.adapted_teacher.carl.json
for path in "$TEACHER" "$DENSE"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done
python scripts/build_carl_targets.py \
  --manifest-in "$DENSE" --manifest-out "$OUT" --teacher "$TEACHER" \
  --out-dir data/carl_ted2_adapted --batch-size "${TARGET_BATCH_SIZE:-16}" --resume
