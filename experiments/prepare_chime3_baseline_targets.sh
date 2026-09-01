#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TEACHER="${CHIME3_ADAPTED_TEACHER:-nemo_experiments/teacher-chime3-adapted-s1/best-teacher-chime3-adapted-s1.nemo}"
DENSE=data/chime3_train.adapted_teacher.frame_dense_t1.json
SCTC=data/chime3_train.adapted_teacher.sctc.json
CARL=data/chime3_train.adapted_teacher.carl.json
CARL_DIR=data/carl_chime3_adapted
SCTC_DIR=data/sctc_chime3_adapted

for path in "$TEACHER" "$DENSE"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

expected=$(wc -l < "$DENSE")
[[ "$expected" -eq 8738 ]] || {
  echo "unexpected CHiME train size: $expected (expected 8738)" >&2
  exit 1
}

if [[ ! -f "$SCTC" ]] || [[ $(wc -l < "$SCTC") -ne "$expected" ]]; then
  python scripts/build_sctc_targets.py \
    --manifest-in "$DENSE" --manifest-out "$SCTC" \
    --out-dir "$SCTC_DIR" --teacher "$TEACHER" \
    --reuse-frame-posterior --resume
fi

if [[ ! -f "$CARL" ]] || [[ $(wc -l < "$CARL") -ne "$expected" ]] || \
   [[ ! -f "$CARL_DIR/teacher_classifier.pt" ]]; then
  python scripts/build_carl_targets.py \
    --manifest-in "$DENSE" --manifest-out "$CARL" --teacher "$TEACHER" \
    --out-dir "$CARL_DIR" --batch-size "${TARGET_BATCH_SIZE:-16}" --resume
fi

for path in "$SCTC" "$CARL" "$CARL_DIR/teacher_classifier.pt"; do
  [[ -f "$path" ]] || { echo "target preparation failed: $path" >&2; exit 1; }
done
[[ $(wc -l < "$SCTC") -eq "$expected" ]]
[[ $(wc -l < "$CARL") -eq "$expected" ]]

echo "[ok] CHiME baseline targets: $expected utterances"
