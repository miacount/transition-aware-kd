#!/usr/bin/env bash
# Combined (logit + token_avg) KD sweep for 8x student.
# Grid: logit_w in {5, 10} x tavg_w in {5, 10}
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.combined.json
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

run_train() {
  local name="$1"; local lw="$2"; local tw="$3"
  echo ""
  echo "════════════════════════════════════════"
  echo "  TRAIN  $name  (logit_w=$lw  tavg_w=$tw)"
  echo "════════════════════════════════════════"
  bash experiments/train.sh \
    --name "$name" \
    --manifest "$MANIFEST" \
    --config student_sub8 \
    --kd-mode combined \
    --kd-weight "$lw" \
    --token-avg-weight "$tw"
}

run_eval() {
  local name="$1"; local lw="$2"; local tw="$3"
  local ckpt_dir
  ckpt_dir=$(ls -dt nemo_experiments/"$name"/*/  2>/dev/null | head -1)checkpoints
  local best_ckpt
  best_ckpt=$(ls "$ckpt_dir"/*.ckpt 2>/dev/null \
    | grep -v "\-last\." \
    | sort -t= -k2 -n \
    | head -1)
  if [[ -z "$best_ckpt" ]]; then
    echo "[eval] no checkpoint found for $name, skipping." >&2
    return
  fi
  echo ""
  echo "════════════════════════════════════════"
  echo "  EVAL   $name  (logit_w=$lw  tavg_w=$tw)"
  echo "════════════════════════════════════════"
  bash experiments/eval.sh \
    --config configs/student_sub8.yaml \
    --ckpt "$best_ckpt" \
    --name "sub8-combined-l${lw}-t${tw}"
}

for lw in 5 10; do
  for tw in 5 10; do
    name="student-sub8-combined-l${lw}-t${tw}-clean"
    run_train "$name" "$lw" "$tw"
    run_eval  "$name" "$lw" "$tw"
  done
done

echo ""
echo "=== Results ==="
echo "--- sub8 baselines ---"
grep -E "test_clean|test_other" analysis/eval_sub8-no-kd.txt          2>/dev/null | sed 's/^/  no-kd       : /'
grep -E "test_clean|test_other" analysis/eval_sub8-logit-t1-w10.0.txt 2>/dev/null | sed 's/^/  logit-w10   : /'
grep -E "test_clean|test_other" analysis/eval_sub8-token-avg-w10.txt  2>/dev/null | sed 's/^/  tavg-w10    : /'
echo "--- sub8 combined sweep ---"
for lw in 5 10; do
  for tw in 5 10; do
    f="analysis/eval_sub8-combined-l${lw}-t${tw}.txt"
    [[ -f "$f" ]] && grep -E "test_clean|test_other" "$f" | sed "s/^/  l${lw}-t${tw}      : /"
  done
done
echo "--- 4x reference ---"
grep -E "test_clean|test_other" analysis/eval_no-kd.txt               2>/dev/null | sed 's/^/  4x no-kd    : /'
grep -E "test_clean|test_other" analysis/eval_token-avg-w15.txt       2>/dev/null | sed 's/^/  4x tavg-w15 : /'
