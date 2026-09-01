#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_TEACHER=/root/.cache/torch/NeMo/NeMo_2.4.1/stt_en_conformer_ctc_small/5d2d8e5b2b5adb8f5091363c6ba19c55/stt_en_conformer_ctc_small.nemo
TEACHER="${LBS_TEACHER_NEMO:-$DEFAULT_TEACHER}"
DENSE=data/train_clean_100.small_teacher.frame_dense_t1.json
OUT=data/train_clean_100.small_teacher.carl.json

for path in "$TEACHER" "$DENSE"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

python scripts/build_carl_targets.py \
  --manifest-in "$DENSE" --manifest-out "$OUT" --teacher "$TEACHER" \
  --out-dir data/carl_lbs_small \
  --batch-size "${TARGET_BATCH_SIZE:-16}" --resume

[[ -f data/carl_lbs_small/teacher_classifier.pt ]] || {
  echo "missing LS CARL classifier artifact" >&2; exit 1;
}
echo "[done] LS CARL/FPKD feature targets"
