#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TEACHER="${CHIME3_ADAPTED_TEACHER:-nemo_experiments/teacher-chime3-adapted-s1/best-teacher-chime3-adapted-s1.nemo}"
PLAIN=data/chime3_train_enhanced.json
DENSE=data/chime3_train.adapted_teacher.frame_dense_t1.json
MASS=data/chime3_train.adapted_teacher.d6_mass3_ntdk_m32.json

for path in "$TEACHER" "$PLAIN"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

echo "=== [1/2] CHiME-3 adapted-teacher dense posterior T=1 ==="
python scripts/build_kd_targets.py \
  --mode frame_dense --manifest_in "$PLAIN" --manifest_out "$DENSE" \
  --out_dir data/frame_dense_chime3_adapted_t1 --teacher "$TEACHER" \
  --temperature 1.0 --batch_size "${TARGET_BATCH_SIZE:-16}" --resume

echo "=== [2/2] CHiME-3 Mass3+NTDK targets ==="
python scripts/build_span_kd_targets.py \
  --manifest-in "$DENSE" --manifest-out "$MASS" \
  --out-dir data/span_mass3_ntdk_chime3_adapted_m32 --teacher "$TEACHER" \
  --teacher-only --teacher-blank-penalty 6 --top-k 8 --dark-top-m 32 \
  --reuse-frame-posterior --resume

echo "=== CHiME-3 target preparation complete ==="
