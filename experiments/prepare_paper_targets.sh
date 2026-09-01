#!/usr/bin/env bash
set -euo pipefail

SCOPE="${1:-all}"
case "$SCOPE" in
  lbs|ted3|all) ;;
  *) echo "usage: bash experiments/prepare_paper_targets.sh [lbs|ted3|all]" >&2; exit 2 ;;
esac

build_dense() {
  local input="$1" output="$2" out_dir="$3"
  python scripts/build_kd_targets.py \
    --mode frame_dense \
    --manifest_in "$input" \
    --manifest_out "$output" \
    --out_dir "$out_dir" \
    --temperature 1.0 \
    --batch_size "${TARGET_BATCH_SIZE:-16}" \
    --resume
}

if [[ "$SCOPE" == "lbs" || "$SCOPE" == "all" ]]; then
  echo "=== [targets] LibriSpeech full posterior T=1 ==="
  build_dense \
    data/train_clean_100.json \
    data/train_clean_100.small_teacher.frame_dense_t1.json \
    data/frame_dense_small_t1

  [[ -f data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json ]] || {
    echo "missing fixed LibriSpeech Mass3+NTDK manifest" >&2; exit 1;
  }
  [[ -f data/train_clean_100.sctc.json ]] || {
    echo "missing LibriSpeech S-CTC manifest" >&2; exit 1;
  }
fi


if [[ "$SCOPE" == "ted3" || "$SCOPE" == "all" ]]; then
  echo "=== [targets] TED-LIUM3 full 454h posterior T=1 ==="
  build_dense \
    data/tedlium3_train.json \
    data/tedlium3_train.small_teacher.frame_dense_t1.json \
    data/frame_dense_tedlium3_t1

  echo "=== [targets] TED-LIUM3 Mass3+NTDK from cached posterior ==="
  python scripts/build_span_kd_targets.py \
    --manifest-in data/tedlium3_train.small_teacher.frame_dense_t1.json \
    --manifest-out data/tedlium3_train.teacher_only_d6_mass3_ntdk_m32.json \
    --out-dir data/span_mass3_ntdk_tedlium3_m32 \
    --teacher-only --teacher-blank-penalty 6 --top-k 8 --dark-top-m 32 \
    --reuse-frame-posterior --resume

  echo "=== [targets] TED-LIUM3 S-CTC occupancy from cached posterior ==="
  python scripts/build_sctc_targets.py \
    --manifest-in data/tedlium3_train.small_teacher.frame_dense_t1.json \
    --manifest-out data/tedlium3_train.sctc.json \
    --out-dir data/sctc_tedlium3 \
    --reuse-frame-posterior --resume
fi

echo "=== target preparation complete: $SCOPE ==="
