#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TEACHER="${TED3_ADAPTED_TEACHER:-nemo_experiments/teacher-ted3-adapted-s1/best-teacher-ted3-adapted-s1.nemo}"
PLAIN=data/tedlium3_train.json
DENSE=data/tedlium3_train.adapted_teacher.frame_dense_t1.json
MASS=data/tedlium3_train.adapted_teacher.d6_mass3_ntdk_m32.json
SCTC=data/tedlium3_train.adapted_teacher.sctc.json

for path in "$TEACHER" "$PLAIN"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

echo "=== [adapted targets 1/3] TED3 dense posterior T=1 ==="
python scripts/build_kd_targets.py \
  --mode frame_dense --manifest_in "$PLAIN" --manifest_out "$DENSE" \
  --out_dir data/frame_dense_ted3_adapted_t1 --teacher "$TEACHER" \
  --temperature 1.0 --batch_size "${TARGET_BATCH_SIZE:-16}" --resume

echo "=== [adapted targets 2/3] TED3 Mass3+NTDK ==="
python scripts/build_span_kd_targets.py \
  --manifest-in "$DENSE" --manifest-out "$MASS" \
  --out-dir data/span_mass3_ntdk_ted3_adapted_m32 --teacher "$TEACHER" \
  --teacher-only --teacher-blank-penalty 6 --top-k 8 --dark-top-m 32 \
  --reuse-frame-posterior --resume

echo "=== [adapted targets 3/3] TED3 S-CTC ==="
python scripts/build_sctc_targets.py \
  --manifest-in "$DENSE" --manifest-out "$SCTC" \
  --out-dir data/sctc_ted3_adapted --teacher "$TEACHER" \
  --reuse-frame-posterior --resume

echo "=== adapted-teacher target preparation complete ==="
