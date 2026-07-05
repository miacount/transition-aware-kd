#!/usr/bin/env bash
# Logit-KD temperature sweep: T=2 (targets exist), T=4, T=8 (need build)
# Purpose: show token-avg improvement is NOT explained by distribution softening.
# T=1 w=1,5,10 already done. T=2 manifest already built.
set -euo pipefail

MANIFEST_IN=data/train_clean_100.json
BASE_MANIFEST=data/train_clean_100.small_teacher.frame_top8

build_targets() {
  local T="$1"
  local OUT="${BASE_MANIFEST}_t${T}.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists"
    return
  fi
  echo ""
  echo "════════════════════════════════════════"
  echo "  BUILD  frame_topk  T=$T"
  echo "════════════════════════════════════════"
  python scripts/build_kd_targets.py \
    --mode frame_topk \
    --manifest_in "$MANIFEST_IN" \
    --manifest_out "$OUT" \
    --temperature "$T" \
    --top_k 8 \
    --batch_size 32
}

run_train() {
  local T="$1"; local W="$2"
  local name="student-logit-t${T}-w${W}-clean"
  echo ""
  echo "════════════════════════════════════════"
  echo "  TRAIN  $name"
  echo "════════════════════════════════════════"
  bash experiments/train.sh \
    --name "$name" \
    --manifest "${BASE_MANIFEST}_t${T}.json" \
    --kd-mode logit \
    --kd-weight "$W" \
    --temperature "$T"
}

run_eval() {
  local T="$1"; local W="$2"
  local name="student-logit-t${T}-w${W}-clean"
  local ckpt_dir
  ckpt_dir=$(ls -dt nemo_experiments/"$name"/*/  2>/dev/null | head -1)checkpoints
  local best_ckpt
  best_ckpt=$(ls "$ckpt_dir"/*.ckpt 2>/dev/null \
    | grep -v "\-last\." \
    | sort -t= -k2 -n \
    | head -1)
  [[ -z "$best_ckpt" ]] && { echo "[eval] no ckpt for $name" >&2; return; }
  echo ""
  echo "════════════════════════════════════════"
  echo "  EVAL   $name"
  echo "════════════════════════════════════════"
  bash experiments/eval.sh \
    --ckpt "$best_ckpt" \
    --name "logit-t${T}-w${W}"
}

# T=2 manifest already exists — skip build
# T=4, T=8 need to be built
for T in 4 8; do
  build_targets "$T"
done

# train + eval: T in {2,4,8}, W in {5,10}
for T in 2 4 8; do
  for W in 5 10; do
    run_train "$T" "$W"
    run_eval  "$T" "$W"
  done
done

echo ""
echo "=== Temperature sweep results ==="
echo "--- logit-KD ---"
for T in 1 2 4 8; do
  for W in 1 5 10; do
    f="analysis/eval_logit-t${T}-w${W}.txt"
    [[ -f "$f" ]] && grep -E "test_clean|test_other" "$f" | sed "s/^/  T=$T w=$W: /"
  done
done
echo "--- token-avg (reference) ---"
grep -E "test_clean|test_other" analysis/eval_token-avg-w15.txt 2>/dev/null | sed 's/^/  tavg-w15: /'
