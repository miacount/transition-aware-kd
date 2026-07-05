#!/usr/bin/env bash
# Token-avg KD sweep for 8x student (w=5, 10, 15, 20)
# Reuses 4x teacher token_avg targets — enc_len/teacher_frames scaling handles mismatch.
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.token_avg.json
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

run_train() {
  local name="$1"; local weight="$2"
  echo ""
  echo "════════════════════════════════════════"
  echo "  TRAIN  $name  (w=$weight)"
  echo "════════════════════════════════════════"
  bash experiments/train.sh \
    --name "$name" \
    --manifest "$MANIFEST" \
    --config student_sub8 \
    --kd-mode token_avg \
    --kd-weight "$weight"
}

run_eval() {
  local name="$1"; local weight="$2"
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
  echo "  EVAL   $name  (w=$weight)"
  echo "════════════════════════════════════════"
  bash experiments/eval.sh \
    --config configs/student_sub8.yaml \
    --ckpt "$best_ckpt" \
    --name "sub8-token-avg-w${weight}"
}

for weight in 5 10 15 20; do
  name="student-sub8-token-avg-w${weight}-clean"
  run_train "$name" "$weight"
  run_eval  "$name" "$weight"
done

echo ""
echo "=== Results ==="
echo "--- baseline ---"
grep -E "test_clean|test_other" analysis/eval_sub8-no-kd.txt 2>/dev/null | sed 's/^/  sub8 no-kd: /'
grep -E "test_clean|test_other" analysis/eval_sub8-logit-t1-w10.0.txt 2>/dev/null | sed 's/^/  sub8 logit-w10: /'
echo "--- token-avg (4x student, best) ---"
grep -E "test_clean|test_other" analysis/eval_token-avg-w15.txt 2>/dev/null | sed 's/^/  4x token-avg-w15: /'
echo "--- sub8 token-avg sweep ---"
for weight in 5 10 15 20; do
  f="analysis/eval_sub8-token-avg-w${weight}.txt"
  [[ -f "$f" ]] && grep -E "test_clean|test_other" "$f" | sed "s/^/  w=$weight: /"
done
