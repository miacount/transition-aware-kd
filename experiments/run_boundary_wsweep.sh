#!/usr/bin/env bash
# Counterfactual Boundary-KD: kd_weight sweep on cell 2 (the proposal).
#
# The boundary loss is batch-normalised over residual mass, a different scale
# from span-KD's occupancy pooling, so span-KD's w=20 does NOT carry over. This
# sweep finds the working range before committing to the 4-cell comparison.
#
# ANCHOR: span_b1_s1 = 12.89 / 31.17 (LS test clean/other, seed 1, current code).
# Boundary-KD should land near it while being far more minimal — not necessarily
# beat it. Watch train/bd_weight_sum and train/bd_res_frames_per_batch in wandb
# to sanity-check the loss scale on the first run.
#
# Sequential on one GPU. Launch inside your own tmux, AFTER the target build:
#   python scripts/build_boundary_kd_targets.py \
#     --manifest-in data/train_clean_100.json \
#     --manifest-out data/train_clean_100.boundary_d6.json \
#     --out-dir boundary_kd_d6 --delta 6.0
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST="data/train_clean_100.boundary_d6.json"
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST — run the build first" >&2; exit 1; }

echo "===== validity gate: boundary_kd loss unit test ====="
python scripts/test_boundary_kd.py || { echo "UNIT TEST FAILED — aborting"; exit 1; }

best_ckpt() {  # lowest val_wer, excluding -last
  ls nemo_experiments/"$1"/*/checkpoints/*.ckpt 2>/dev/null | grep -v last \
    | while read -r f; do
        wer=$(basename "$f" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
        echo "$wer $f"
      done | sort -n | head -1 | cut -d' ' -f2-
}

run() {
  local w="$1"; local name="bd-cell2-w${w}"
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  $(date '+%F %T')  $name"
  echo "════════════════════════════════════════════════════════"
  bash experiments/train.sh --config student_base --name "$name" \
      --manifest "$MANIFEST" --kd-mode boundary_kd --kd-weight "$w" \
      --boundary-cell 2 --seed 1
  local ck; ck=$(best_ckpt "$name")
  if [[ -z "$ck" ]]; then echo "[warn] no ckpt for $name, skipping eval" >&2; return 0; fi
  echo "[eval] $name  <-  $ck"
  bash experiments/eval.sh --ckpt "$ck" --name "$name"
}

for W in 5 10 20 40; do
  run "$W"
done

echo ""
echo "===== sweep complete ====="
echo "results: analysis/eval_bd-cell2-w{5,10,20,40}.txt   (anchor 12.89/31.17)"
grep -H -E "test_clean|test_other" analysis/eval_bd-cell2-w*.txt 2>/dev/null || true
