#!/usr/bin/env bash
# Token-avg KD high-weight sweep (w=15, 20)
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
    --ckpt "$best_ckpt" \
    --name "token-avg-w${weight}"
}

for weight in 15 20; do
  name="student-token-avg-w${weight}-clean"
  run_train "$name" "$weight"
  run_eval  "$name" "$weight"
done

echo ""
echo "=== Results ==="
for weight in 10.0 15 20; do
  f="analysis/eval_token-avg-w${weight}.txt"
  if [[ -f "$f" ]]; then
    echo "w=$weight:"
    grep -E "test_clean|test_other" "$f" || true
  fi
done
