#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

# Equal-budget, dev-only loss-scale calibration for Citrinet-144 / Citrinet-256.
# Three candidates per KD method. Candidate selection uses dev-clean val_wer only.
# Standard methods receive 30 branch epochs per candidate. CR-CTC receives 15
# epochs because each step executes two encoder forwards. Multi-stage deterministic
# prefixes are cached once, then all three scale candidates branch from the same
# checkpoint. No dev-other or test split is evaluated in this script.

CONFIG=student_citrinet144
PLAIN=data/train_clean_100.json
DENSE=data/train_clean_100.citrinet256.frame_dense_t1.json
MASS=data/train_clean_100.citrinet256.d6_mass3_ntdk_m32.json
SCTC=data/train_clean_100.citrinet256.sctc.json
CARL=data/train_clean_100.citrinet256.carl.json
CLASSIFIER=data/train_clean_100.citrinet256.carl/teacher_classifier.pt
WARM_NAME=arch-citrinet144-t256-lbs-cal100-shared-ctc10-s1
PREFIX=arch-citrinet144-t256-lbs-tune30
CANDIDATES=analysis/citrinet144_tune30_candidates.tsv
SELECTION=analysis/citrinet144_tune30_selection.tsv
TAGS=(lo mid hi)

COMMON=(
  --config "$CONFIG"
  --train-batch-size 64
  --accumulate-grad-batches 8
  --lr 0.025
  --dev-only
  --save-top-k 1
)

require_assets() {
  local file
  for file in "$PLAIN" "$DENSE" "$MASS" "$SCTC" "$CARL" "$CLASSIFIER"; do
    require_file "$file"
  done
}

warm_ckpt() {
  local ckpt
  ckpt=$(latest_last_ckpt "$WARM_NAME")
  [[ -n "$ckpt" ]] || { echo "missing shared warm-start checkpoint: $WARM_NAME" >&2; exit 1; }
  printf '%s\n' "$ckpt"
}

scale_values() {
  local method="$1" tag="$2"
  case "$method:$tag" in
    atdk:lo) echo "10 3" ;;
    atdk:mid) echo "20 6" ;;
    atdk:hi) echo "40 12" ;;
    vanilla:lo|kdbe:lo|symmetric:lo) echo "0.125" ;;
    vanilla:mid|kdbe:mid|symmetric:mid) echo "0.25" ;;
    vanilla:hi|kdbe:hi|symmetric:hi) echo "0.50" ;;
    guided:lo) echo "2.5" ;;
    guided:mid) echo "5" ;;
    guided:hi) echo "10" ;;
    sctc:lo) echo "0.96" ;;
    sctc:mid) echo "0.98" ;;
    sctc:hi) echo "0.99" ;;
    crctc:lo) echo "0.25" ;;
    crctc:mid) echo "0.5" ;;
    crctc:hi) echo "1.0" ;;
    fpkd:lo|carl:lo) echo "0.5" ;;
    fpkd:mid|carl:mid) echo "1.0" ;;
    fpkd:hi|carl:hi) echo "2.0" ;;
    *) echo "unknown method/tag: $method/$tag" >&2; exit 2 ;;
  esac
}

run_single_stage_tuning() {
  local warm method tag name values primary dark
  warm=$(warm_ckpt)
  for method in atdk vanilla kdbe symmetric guided; do
    for tag in "${TAGS[@]}"; do
      name="${PREFIX}-${method}-${tag}-s1"
      values=$(scale_values "$method" "$tag")
      case "$method" in
        atdk)
          read -r primary dark <<<"$values"
          run_paper_train "$name" 30 --manifest "$MASS" --kd-mode span_kd \
            --kd-weight "$primary" --span-primary-mode mass3 --span-ntdk-weight "$dark" \
            --init-ckpt "$warm" --warmup-steps 300 "${COMMON[@]}"
          ;;
        vanilla)
          run_paper_train "$name" 30 --manifest "$DENSE" --kd-mode logit \
            --temperature 1 --logit-reduction utterance_sum --blank-mode none \
            --kd-lambda "$values" --init-ckpt "$warm" --warmup-steps 300 "${COMMON[@]}"
          ;;
        kdbe)
          run_paper_train "$name" 30 --manifest "$DENSE" --kd-mode logit \
            --temperature 1 --logit-reduction utterance_sum --blank-mode elimination \
            --kd-lambda "$values" --init-ckpt "$warm" --warmup-steps 300 "${COMMON[@]}"
          ;;
        symmetric)
          run_paper_train "$name" 30 --manifest "$DENSE" --kd-mode logit \
            --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric --blank-n 4 \
            --kd-lambda "$values" --init-ckpt "$warm" --warmup-steps 300 "${COMMON[@]}"
          ;;
        guided)
          run_paper_train "$name" 30 --manifest "$DENSE" --kd-mode guided \
            --temperature 1 --kd-weight "$values" \
            --init-ckpt "$warm" --warmup-steps 300 "${COMMON[@]}"
          ;;
      esac
    done
  done
}

run_sctc_tuning() {
  local warm tag scale kd_name kd_ckpt ft_name
  warm=$(warm_ckpt)
  for tag in "${TAGS[@]}"; do
    scale=$(scale_values sctc "$tag")
    kd_name="${PREFIX}-sctc-${tag}-kd-s1"
    ft_name="${PREFIX}-sctc-${tag}-ctcft-s1"
    run_paper_train "$kd_name" 25 --manifest "$SCTC" --kd-mode sctc \
      --kd-lambda "$scale" --init-ckpt "$warm" --warmup-steps 250 "${COMMON[@]}"
    kd_ckpt=$(latest_last_ckpt "$kd_name")
    [[ -n "$kd_ckpt" ]] || { echo "missing S-CTC tuning checkpoint: $kd_name" >&2; exit 1; }
    run_paper_train "$ft_name" 5 --manifest "$PLAIN" --kd-mode none \
      --init-ckpt "$kd_ckpt" --warmup-steps 50 "${COMMON[@]}"
  done
}

run_crctc_tuning() {
  local warm tag scale name
  warm=$(warm_ckpt)
  for tag in "${TAGS[@]}"; do
    scale=$(scale_values crctc "$tag")
    name="${PREFIX}-crctc-${tag}-s1"
    run_paper_train "$name" 15 --manifest "$PLAIN" --kd-mode cr_ctc \
      --cr-ctc-weight "$scale" --cr-ctc-warm-step 150 --cr-ctc-time-factor 1.5 \
      --init-ckpt "$warm" --warmup-steps 150 "${COMMON[@]}"
  done
}

run_fpkd_tuning() {
  local warm dfkd_name dfkd_ckpt frkd_name frkd_ckpt tag scale name
  warm=$(warm_ckpt)
  dfkd_name="${PREFIX}-fpkd-shared-dfkd-s1"
  frkd_name="${PREFIX}-fpkd-shared-frkd-s1"
  run_paper_train "$dfkd_name" 5 --manifest "$DENSE" --kd-mode fpkd_dfkd \
    --fpkd-temp 1 --init-ckpt "$warm" --warmup-steps 50 "${COMMON[@]}"
  dfkd_ckpt=$(latest_last_ckpt "$dfkd_name")
  [[ -n "$dfkd_ckpt" ]] || { echo "missing FPKD DFKD tuning checkpoint" >&2; exit 1; }
  run_paper_train "$frkd_name" 5 --manifest "$CARL" --kd-mode fpkd_frkd \
    --carl-teacher-dim 640 --init-ckpt "$dfkd_ckpt" --warmup-steps 50 "${COMMON[@]}"
  frkd_ckpt=$(latest_last_ckpt "$frkd_name")
  [[ -n "$frkd_ckpt" ]] || { echo "missing FPKD FRKD tuning checkpoint" >&2; exit 1; }
  for tag in "${TAGS[@]}"; do
    scale=$(scale_values fpkd "$tag")
    name="${PREFIX}-fpkd-${tag}-pkd-s1"
    run_paper_train "$name" 20 --manifest "$DENSE" --kd-mode fpkd_pkd \
      --fpkd-bkl-weight "$scale" --fpkd-nbf-weight "$scale" --fpkd-temp 1 \
      --init-ckpt "$frkd_ckpt" --warmup-steps 200 "${COMMON[@]}"
  done
}

run_carl_tuning() {
  local warm feature_name feature_ckpt tag scale name
  warm=$(warm_ckpt)
  feature_name="${PREFIX}-carl-shared-feature-s1"
  run_paper_train "$feature_name" 5 --manifest "$CARL" --kd-mode carl_feature \
    --carl-teacher-dim 640 --init-ckpt "$warm" --warmup-steps 50 "${COMMON[@]}"
  feature_ckpt=$(latest_last_ckpt "$feature_name")
  [[ -n "$feature_ckpt" ]] || { echo "missing CARL feature tuning checkpoint" >&2; exit 1; }
  for tag in "${TAGS[@]}"; do
    scale=$(scale_values carl "$tag")
    name="${PREFIX}-carl-${tag}-full-s1"
    run_paper_train "$name" 25 --manifest "$CARL" --kd-mode carl \
      --carl-classifier "$CLASSIFIER" --carl-teacher-dim 640 \
      --carl-alpha "$scale" --carl-gamma "$scale" --carl-lambda "$scale" \
      --init-ckpt "$feature_ckpt" --warmup-steps 250 "${COMMON[@]}"
  done
}

candidate_name() {
  local method="$1" tag="$2"
  case "$method" in
    sctc) echo "${PREFIX}-sctc-${tag}-ctcft-s1" ;;
    fpkd) echo "${PREFIX}-fpkd-${tag}-pkd-s1" ;;
    carl) echo "${PREFIX}-carl-${tag}-full-s1" ;;
    *) echo "${PREFIX}-${method}-${tag}-s1" ;;
  esac
}

summarize() {
  local method tag name ckpt wer value best_line
  mkdir -p analysis
  printf 'method\ttag\tscale\tdev_clean_wer\tcheckpoint\n' > "$CANDIDATES"
  printf 'method\ttag\tscale\tdev_clean_wer\tcheckpoint\n' > "$SELECTION"
  for method in atdk vanilla kdbe symmetric guided sctc crctc fpkd carl; do
    for tag in "${TAGS[@]}"; do
      name=$(candidate_name "$method" "$tag")
      ckpt=$(best_ckpt "$name")
      [[ -n "$ckpt" ]] || { echo "missing tuning checkpoint: $name" >&2; exit 1; }
      wer=$(basename "$ckpt" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
      value=$(scale_values "$method" "$tag")
      printf '%s\t%s\t%s\t%s\t%s\n' "$method" "$tag" "$value" "$wer" "$ckpt" >> "$CANDIDATES"
    done
    best_line=$(awk -F '\t' -v m="$method" '$1==m {print}' "$CANDIDATES" | sort -t $'\t' -k4,4n | head -n1)
    printf '%s\n' "$best_line" >> "$SELECTION"
  done
  echo "=== tuning selections ==="
  column -t -s $'\t' "$SELECTION" 2>/dev/null || tee /dev/stderr < "$SELECTION" >/dev/null
}

require_assets
run_single_stage_tuning
run_sctc_tuning
run_crctc_tuning
run_fpkd_tuning
run_carl_tuning
summarize
