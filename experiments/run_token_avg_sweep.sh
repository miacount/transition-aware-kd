#!/usr/bin/env bash
# Token-avg KD weight sweep (5 runs, sequential).
# Waits for manifest to be ready, then trains + evals each run.
# Usage: bash experiments/run_token_avg_sweep.sh
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.token_avg.json

# ── wait for manifest ─────────────────────────────────────────────────────────
if [[ ! -f "$MANIFEST" ]]; then
  echo "[wait] $MANIFEST not found yet — waiting for build to complete..."
  until [[ -f "$MANIFEST" ]]; do sleep 30; done
  echo "[wait] manifest ready."
fi

# ── helpers ───────────────────────────────────────────────────────────────────
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
  # find best checkpoint (lowest val_wer, non-last)
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

# ── sweep ─────────────────────────────────────────────────────────────────────
for weight in 0.5 1.0 2.0 5.0 10.0; do
  name="student-token-avg-w${weight}-clean"
  run_train "$name" "$weight"
  run_eval  "$name" "$weight"
done

echo ""
echo "All token_avg sweep runs complete."
echo ""
echo "=== Results summary ==="
for weight in 0.5 1.0 2.0 5.0 10.0; do
  f="analysis/eval_token-avg-w${weight}.txt"
  if [[ -f "$f" ]]; then
    echo "w=$weight:"
    grep -E "test_clean|test_other" "$f" || true
  fi
done
