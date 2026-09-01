#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LS_DENSE=data/train_clean_100.small_teacher.frame_dense_t1.json
CH_DENSE=data/chime3_train.adapted_teacher.frame_dense_t1.json
CH_TEACHER=nemo_experiments/teacher-chime3-adapted-s1/best-teacher-chime3-adapted-s1.nemo

for path in "$LS_DENSE" "$CH_DENSE" "$CH_TEACHER"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done

echo "=== [1/2] LibriSpeech delta=9 Mass3+NTDK, M=32 ==="
python scripts/build_span_kd_targets.py \
  --manifest-in "$LS_DENSE" \
  --manifest-out data/train_clean_100.teacher_only_d9_mass3_ntdk_m32.json \
  --out-dir data/span_mass3_ntdk_lbs_d9_m32 \
  --teacher stt_en_conformer_ctc_small --teacher-only \
  --teacher-blank-penalty 9 --top-k 8 --dark-top-m 32 \
  --reuse-frame-posterior --resume

echo "=== [2/2] CHiME-3 delta=9 Mass3+NTDK, M=32 ==="
python scripts/build_span_kd_targets.py \
  --manifest-in "$CH_DENSE" \
  --manifest-out data/chime3_train.adapted_teacher.d9_mass3_ntdk_m32.json \
  --out-dir data/span_mass3_ntdk_chime3_adapted_d9_m32 \
  --teacher "$CH_TEACHER" --teacher-only \
  --teacher-blank-penalty 9 --top-k 8 --dark-top-m 32 \
  --reuse-frame-posterior --resume

echo "=== delta=9 targets complete ==="
