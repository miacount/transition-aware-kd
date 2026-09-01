#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

SCOPE="${1:-lbs}"
BEAM_WIDTH="${BEAM_WIDTH:-16}"
mkdir -p analysis

run_decode() {
  local label="$1" exp_name="$2" config="$3"
  shift 3
  local ckpt log
  ckpt=$(best_ckpt "$exp_name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint for $exp_name" >&2; exit 1; }
  log="analysis/beam${BEAM_WIDTH}_paper_${label}.txt"
  if [[ "${FORCE_EVAL:-0}" != "1" && -s "$log" ]] \
      && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  if [[ "${FORCE_EVAL:-0}" != "1" && -s "$log" ]]; then
    echo "=== [refresh] stale log (checkpoint changed): $log ==="
  fi
  echo "=== [beam${BEAM_WIDTH}] $label: $ckpt ==="
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" --name "$label" --config "$config" \
    --beam-width "$BEAM_WIDTH" "$@" | tee "$log"
}

case "$SCOPE" in
  lbs)
    MANIFESTS=(
      --manifest dev_clean=data/dev_clean.json
      --manifest dev_other=data/dev_other.json
      --manifest test_clean=data/test_clean.json
      --manifest test_other=data/test_other.json
    )
    run_decode lbs_nokd student-no-kd configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_vanilla_full_l025 paper-lbs-vanilla-full-l025-s1 configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_kdbe_full_l025 paper-lbs-kdbe-full-l025-s1 configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_symmetric_full_l025_n4 paper-lbs-symmetric-full-l025-n4-s1 configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_guided_exact paper-lbs-guided-exact-w1-s1 configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_sctc_ft paper-lbs-sctc-l1-ctcft-s1 configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_crctc_stable_fair50 cr_ctc_w02_fair50 configs/student_base.yaml "${MANIFESTS[@]}"
    run_decode lbs_mass25_ntdk8 span-d6-mass3-w25-ntdk-a8-s1 configs/student_base.yaml "${MANIFESTS[@]}"
    ;;
  ted2)
    MANIFESTS=(
      --manifest ted2_test=data/tedlium2_test.json
    )
    run_decode ted2_nokd paper-ted2-no-kd-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_vanilla_full_l025 paper-ted2-lsfrozen-vanilla-full-l025-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_kdbe_full_l025 paper-ted2-lsfrozen-kdbe-full-l025-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_symmetric_full_l025_n4 paper-ted2-lsfrozen-symmetric-full-l025-n4-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_guided_exact paper-ted2-guided-exact-w1-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_sctc_ft paper-ted2-sctc-l1-ctcft-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_crctc_lsfrozen paper-ted2-crctc-lsfrozen-e50-b16a4-f1p5-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    run_decode ted2_mass25_ntdk8 paper-ted2-mass3-w25-ntdk8-s1 configs/student_base_ted2.yaml "${MANIFESTS[@]}"
    ;;
  ted3)
    MANIFESTS=(
      --manifest ted3_dev=data/tedlium3_dev.json
      --manifest ted3_test=data/tedlium3_test.json
    )
    run_decode ted3_nokd paper-ted3full-no-kd-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_vanilla_full_l09 paper-ted3full-vanilla-full-l09-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_kdbe_full_l09 paper-ted3full-kdbe-full-l09-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_symmetric_full_l1_n2 paper-ted3full-symmetric-full-l1-n2-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_guided_exact paper-ted3full-guided-exact-w1-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_sctc_ft paper-ted3full-sctc-l1-ctcft-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_crctc_fair paper-ted3full-crctc-fair25-b16-mask2p5-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3_mass25_ntdk8 paper-ted3full-mass3-w25-ntdk8-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    ;;
  ted3adapt)
    MANIFESTS=(
      --manifest ted3_dev=data/tedlium3_dev.json
      --manifest ted3_test=data/tedlium3_test.json
    )
    run_decode ted3adapt_vanilla_full_l09 paper-ted3adapt-vanilla-full-l09-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_kdbe_full_l09 paper-ted3adapt-kdbe-full-l09-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_symmetric_full_l1_n2 paper-ted3adapt-symmetric-full-l1-n2-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_guided_exact paper-ted3adapt-guided-exact-w1-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_sctc_ft paper-ted3adapt-sctc-l1-ctcft-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_mass25_ntdk8 paper-ted3adapt-mass3-w25-ntdk8-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    ;;
  ted3ratiocal)
    MANIFESTS=(
      --manifest ted3_dev=data/tedlium3_dev.json
      --manifest ted3_test=data/tedlium3_test.json
    )
    run_decode ted3adapt_ratiocal_vanilla_l0289 paper-ted3adapt-ratiocal-vanilla-l0289-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_ratiocal_kdbe_l0439 paper-ted3adapt-ratiocal-kdbe-l0439-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_ratiocal_symmetric_l0310_n2 paper-ted3adapt-ratiocal-symmetric-l0310-n2-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_ratiocal_guided_w1202 paper-ted3adapt-ratiocal-guided-w1202-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    run_decode ted3adapt_ratiocal_mass3_w16p53_ntdk4p85 paper-ted3adapt-ratiocal-mass3-w16p53-ntdk4p85-s1 configs/student_base_ted3.yaml "${MANIFESTS[@]}"
    ;;
  *) echo "usage: bash experiments/evaluate_paper_main.sh [lbs|ted2|ted3|ted3adapt|ted3ratiocal]" >&2; exit 2 ;;
esac
