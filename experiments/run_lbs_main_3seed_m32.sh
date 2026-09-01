#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/train_clean_100.json
DENSE=data/train_clean_100.small_teacher.frame_dense_t1.json
MASS=data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json
SCTC=data/train_clean_100.sctc.json
BEAM_WIDTH="${BEAM_WIDTH:-16}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p analysis
for path in "$PLAIN" "$DENSE" "$MASS" "$SCTC"; do require_file "$path"; done

experiment_name() {
  local method="$1" seed="$2"
  case "$method:$seed" in
    nokd:1) echo student-no-kd ;;
    vanilla:1) echo paper-lbs-vanilla-full-l025-s1 ;;
    kdbe:1) echo paper-lbs-kdbe-full-l025-s1 ;;
    symmetric:1) echo paper-lbs-symmetric-full-l025-n4-s1 ;;
    guided:1) echo paper-lbs-guided-exact-w1-s1 ;;
    sctc:1) echo paper-lbs-sctc-l1-s1 ;;
    sctc_ft:1) echo paper-lbs-sctc-l1-ctcft-s1 ;;
    crctc:1) echo cr_ctc_w02_fair50 ;;
    ours:1) echo span-d6-mass3-w25-ntdk-a8-s1 ;;
    nokd:*) echo "paper-lbs-nokd-s${seed}" ;;
    vanilla:*) echo "paper-lbs-vanilla-full-l025-s${seed}" ;;
    kdbe:*) echo "paper-lbs-kdbe-full-l025-s${seed}" ;;
    symmetric:*) echo "paper-lbs-symmetric-full-l025-n4-s${seed}" ;;
    guided:*) echo "paper-lbs-guided-exact-w1-s${seed}" ;;
    sctc:*) echo "paper-lbs-sctc-l1-s${seed}" ;;
    sctc_ft:*) echo "paper-lbs-sctc-l1-ctcft-s${seed}" ;;
    crctc:*) echo "paper-lbs-crctc-stable-fair50-s${seed}" ;;
    ours:*) echo "paper-lbs-m32-mass25-ntdk8-s${seed}" ;;
    *) echo "unknown method/seed: $method/$seed" >&2; return 2 ;;
  esac
}

run_seeded_train() {
  local name="$1" epochs="$2" seed="$3"
  shift 3
  if experiment_complete "$name" "$epochs"; then
    echo "=== [reuse] $name seed=$seed epochs=$epochs ==="
    return 0
  fi
  local last version
  last=$(latest_last_ckpt "$name")
  if [[ -n "$last" ]]; then
    version=$(basename "$(dirname "$(dirname "$last")")")
    echo "=== [resume] $name seed=$seed from $last ==="
    bash experiments/train.sh --name "$name" --epochs "$epochs" --seed "$seed" \
      --resume-version "$version" "$@"
  else
    echo "=== [train] $name seed=$seed epochs=$epochs ==="
    bash experiments/train.sh --name "$name" --epochs "$epochs" --seed "$seed" "$@"
  fi
  [[ -n "$(best_ckpt "$name")" ]] || {
    echo "no finite checkpoint after training: $name" >&2; exit 1;
  }
}

train_seed() {
  local seed="$1" name stage_ckpt

  name=$(experiment_name nokd "$seed")
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$PLAIN" --kd-mode none

  name=$(experiment_name vanilla "$seed")
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25

  name=$(experiment_name kdbe "$seed")
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum --blank-mode elimination --kd-lambda 0.25

  name=$(experiment_name symmetric "$seed")
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric \
    --blank-n 4 --kd-lambda 0.25

  name=$(experiment_name guided "$seed")
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$DENSE" --kd-mode guided \
    --temperature 1 --kd-weight 1

  name=$(experiment_name sctc "$seed")
  run_seeded_train "$name" 80 "$seed" \
    --config student_base --manifest "$SCTC" --kd-mode sctc --kd-lambda 1
  stage_ckpt=$(best_ckpt "$name")
  [[ -n "$stage_ckpt" ]] || { echo "missing S-CTC stage checkpoint: $name" >&2; exit 1; }
  name=$(experiment_name sctc_ft "$seed")
  run_seeded_train "$name" 20 "$seed" \
    --config student_base --manifest "$PLAIN" --kd-mode none --init-ckpt "$stage_ckpt"

  name=$(experiment_name crctc "$seed")
  run_seeded_train "$name" 50 "$seed" \
    --config student_base --manifest "$PLAIN" --kd-mode cr_ctc \
    --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000 --cr-ctc-time-factor 1.5

  name=$(experiment_name ours "$seed")
  run_seeded_train "$name" 100 "$seed" \
    --config student_base --manifest "$MASS" --kd-mode span_kd \
    --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8
}

evaluate_one() {
  local method="$1" seed="$2" name ckpt log
  name=$(experiment_name "$method" "$seed")
  ckpt=$(best_ckpt "$name")
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $name" >&2; exit 1; }
  log="analysis/beam${BEAM_WIDTH}_main3seed_lbs_${method}_s${seed}.txt"
  if [[ -s "$log" ]] && grep -Fxq "checkpoint=$ckpt" "$log"; then
    echo "=== [reuse] $log ==="
    return 0
  fi
  echo "=== [beam${BEAM_WIDTH}] $method seed=$seed ==="
  python scripts/evaluate_ctc_decoding.py \
    --ckpt "$ckpt" --name "lbs_${method}_s${seed}" \
    --config configs/student_base.yaml --beam-width "$BEAM_WIDTH" \
    --manifest dev_clean=data/dev_clean.json \
    --manifest dev_other=data/dev_other.json \
    --manifest test_clean=data/test_clean.json \
    --manifest test_other=data/test_other.json | tee "$log"
}

for seed in $SEEDS; do
  case "$seed" in 1|2|3) ;; *) echo "SEEDS must contain only 1, 2, 3" >&2; exit 2 ;; esac
  echo "========== MAIN TABLE SEED $seed =========="
  train_seed "$seed"
  for method in nokd vanilla kdbe symmetric guided sctc_ft crctc ours; do
    evaluate_one "$method" "$seed"
  done
done

python scripts/summarize_lbs_main_3seed.py --beam-width "$BEAM_WIDTH"
echo "=== LibriSpeech M=32 main table 3-seed suite complete ==="
