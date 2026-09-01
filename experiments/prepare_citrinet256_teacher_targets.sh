#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SCOPE="${1:-all}"
TARGET_SET="${2:-pilot}"
case "$SCOPE" in lbs|chime3|all) ;; *) echo "usage: $0 [lbs|chime3|all] [pilot|full]" >&2; exit 2 ;; esac
case "$TARGET_SET" in pilot|full) ;; *) echo "usage: $0 [lbs|chime3|all] [pilot|full]" >&2; exit 2 ;; esac
bash experiments/prepare_citrinet_assets.sh

TOKENIZER=tokenizer_citrinet_1024
LBS_TEACHER=stt_en_citrinet_256
CHIME_TEACHER=nemo_experiments/teacher-citrinet256-chime3-adapted-s1/best-teacher-citrinet256-chime3-adapted-s1.nemo
LBS_ARTIFACT=$(find /root/.cache/torch/NeMo -type f -path '*/stt_en_citrinet_256/*.nemo' -print -quit)

build_all_targets() {
  local plain="$1" teacher="$2" teacher_file="$3" prefix="$4"
  local dense="data/${prefix}.frame_dense_t1.json"
  local mass="data/${prefix}.d6_mass3_ntdk_m32.json"
  local sctc="data/${prefix}.sctc.json"
  local carl="data/${prefix}.carl.json"

  python scripts/build_kd_targets.py --mode frame_dense     --manifest_in "$plain" --manifest_out "$dense"     --out_dir "data/${prefix}.frame_dense_t1" --teacher "$teacher"     --temperature 1 --batch_size "${TARGET_BATCH_SIZE:-16}" --resume

  python scripts/build_span_kd_targets.py     --manifest-in "$dense" --manifest-out "$mass"     --out-dir "data/${prefix}.span_mass3_ntdk_m32"     --teacher "$teacher" --tokenizer "$TOKENIZER" --teacher-only     --teacher-blank-penalty 6 --top-k 8 --dark-top-m 32     --reuse-frame-posterior --resume

  [[ "$TARGET_SET" == full ]] || return 0

  python scripts/build_sctc_targets.py     --manifest-in "$dense" --manifest-out "$sctc"     --out-dir "data/${prefix}.sctc" --teacher "$teacher"     --tokenizer "$TOKENIZER" --reuse-frame-posterior --resume

  python scripts/build_carl_targets.py     --manifest-in "$dense" --manifest-out "$carl"     --teacher "$teacher_file" --out-dir "data/${prefix}.carl"     --batch-size "${TARGET_BATCH_SIZE:-16}" --resume
}

if [[ "$SCOPE" == lbs || "$SCOPE" == all ]]; then
  python scripts/evaluate_teacher.py --teacher "$LBS_TEACHER"     --manifest dev_clean=data/dev_clean.json --manifest dev_other=data/dev_other.json     --manifest test_clean=data/test_clean.json --manifest test_other=data/test_other.json     | tee analysis/citrinet256_teacher_lbs.txt
  build_all_targets data/train_clean_100.json "$LBS_TEACHER" "$LBS_ARTIFACT" train_clean_100.citrinet256
fi

if [[ "$SCOPE" == chime3 || "$SCOPE" == all ]]; then
  if [[ ! -f "$CHIME_TEACHER" ]]; then
    python scripts/finetune_teacher_ted3.py --config-name teacher_finetune_citrinet256_chime3
  fi
  python scripts/evaluate_teacher.py --teacher "$CHIME_TEACHER"     --manifest dev_real=data/chime3_dev_real_enhanced.json     --manifest dev_simu=data/chime3_dev_simu_enhanced.json     --manifest eval_real=data/chime3_eval_real_enhanced.json     --manifest eval_simu=data/chime3_eval_simu_enhanced.json     | tee analysis/citrinet256_teacher_chime3.txt
  build_all_targets data/chime3_train_enhanced.json "$CHIME_TEACHER" "$CHIME_TEACHER" chime3_train.citrinet256_adapted
fi
