#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

derive_m4() {
  local source="$1" manifest="$2" out_dir="$3"
  [[ -f "$source" ]] || { echo "missing source manifest: $source" >&2; exit 1; }
  echo "=== [target M=4] $manifest ==="
  python scripts/truncate_ntdk_targets.py \
    --manifest-in "$source" --manifest-out "$manifest" \
    --out-dir "$out_dir" --top-m 4 --resume
}

derive_m4 \
  data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json \
  data/train_clean_100.teacher_only_d6_mass3_ntdk_m4.json \
  data/span_mass3_ntdk_m4
derive_m4 \
  data/chime3_train.adapted_teacher.d6_mass3_ntdk_m32.json \
  data/chime3_train.adapted_teacher.d6_mass3_ntdk_m4.json \
  data/span_mass3_ntdk_chime3_adapted_d6_m4
derive_m4 \
  data/train_clean_100.teacher_only_d0_mass3_ntdk_m32.json \
  data/train_clean_100.teacher_only_d0_mass3_ntdk_m4.json \
  data/span_mass3_ntdk_lbs_d0_m4
derive_m4 \
  data/chime3_train.adapted_teacher.d0_mass3_ntdk_m32.json \
  data/chime3_train.adapted_teacher.d0_mass3_ntdk_m4.json \
  data/span_mass3_ntdk_chime3_adapted_d0_m4
derive_m4 \
  data/train_clean_100.teacher_only_d9_mass3_ntdk_m32.json \
  data/train_clean_100.teacher_only_d9_mass3_ntdk_m4.json \
  data/span_mass3_ntdk_lbs_d9_m4
derive_m4 \
  data/chime3_train.adapted_teacher.d9_mass3_ntdk_m32.json \
  data/chime3_train.adapted_teacher.d9_mass3_ntdk_m4.json \
  data/span_mass3_ntdk_chime3_adapted_d9_m4

echo "=== [target M=4] deterministic permuted NTDK control ==="
python scripts/permute_ntdk_targets.py \
  --manifest-in data/train_clean_100.teacher_only_d6_mass3_ntdk_m4.json \
  --manifest-out data/train_clean_100.teacher_only_d6_mass3_ntdk_m4_permuted.json \
  --out-dir data/span_mass3_ntdk_lbs_d6_m4_permuted \
  --seed 20260818 --resume

echo "=== all M=4 finalization targets complete ==="
