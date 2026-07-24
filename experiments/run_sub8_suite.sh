#!/usr/bin/env bash
# Sub8 generalization suite (roadmap P1-4): reproduce the paper's baseline rows
# on an 8x-subsampling student. Teacher stays sub4 (40ms) while the student runs
# at 80ms: every frame-level method must cross frame rates via the nearest-frame
# resample paths in model.py, while span-KD pools per token — the setup where
# the structural difference between the methods is largest.
#
# All KD hyperparameters are copied verbatim from the corresponding sub4 runs'
# cmd-args.log (kd-be-w10 / kd-sym-n1-w3 / sctc-lambda1(+ctcft) / kd-vanilla-w10 /
# span-kd-teacher-d6-w20.0). Teacher-side target manifests are reused as-is.
#
# Usage:
#   bash experiments/run_sub8_suite.sh              # core 7 runs, sequential
#   SKIP_VANILLA=1 bash experiments/run_sub8_suite.sh
#   RUN_EXTRA=1    bash experiments/run_sub8_suite.sh   # + guided-w1, cr-fair50
set -euo pipefail

SPAN_MANIFEST=data/train_clean_100.teacher_only_d6.json
FRAME_MANIFEST=data/train_clean_100.small_teacher.frame_top8_t2.json
FRAME_T1_MANIFEST=data/train_clean_100.small_teacher.frame_top8_t1.json
SCTC_MANIFEST=data/train_clean_100.sctc.json

for f in "$SPAN_MANIFEST" "$FRAME_MANIFEST" "$SCTC_MANIFEST"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

run() {
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  $(date '+%F %T')  $*"
  echo "════════════════════════════════════════════════════════"
  bash experiments/train.sh "$@"
}

best_ckpt() {  # lowest val_wer, excluding -last
  ls nemo_experiments/"$1"/*/checkpoints/*.ckpt 2>/dev/null | grep -v last \
    | while read -r f; do
        wer=$(basename "$f" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
        echo "$wer $f"
      done | sort -n | head -1 | cut -d' ' -f2-
}

# 1) baseline anchor
run --config student_sub8 --name sub8-no-kd \
    --manifest data/train_clean_100.json --kd-mode none

# 2) ours (= span-kd-teacher-d6-w20.0)
run --config student_sub8 --name sub8-span-d6-w20 \
    --manifest "$SPAN_MANIFEST" --kd-mode span_kd --kd-weight 20

# 3) KD-BE (= kd-be-w10: blank elimination)
run --config student_sub8 --name sub8-kd-be-w10 \
    --manifest "$FRAME_MANIFEST" --kd-mode logit --kd-weight 10 \
    --temperature 2.0 --blank-mode elimination

# 3b) KD-BE + cross-pool: frame-KD given its best shot at cross frame rates —
#     each student frame averages BOTH covering teacher frames, so teacher
#     spikes cannot vanish on the dropped parity (the pooled-target fix).
run --config student_sub8 --name sub8-kd-be-pool-w10 \
    --manifest "$FRAME_MANIFEST" --kd-mode logit --kd-weight 10 \
    --temperature 2.0 --blank-mode elimination --cross-pool

# 4) Sym n=1 (= kd-sym-n1-w3)
run --config student_sub8 --name sub8-sym-n1-w3 \
    --manifest "$FRAME_MANIFEST" --kd-mode logit --kd-weight 3 \
    --temperature 2.0 --blank-mode symmetric --blank-n 1

# 5) S-CTC lambda=1 (= sctc-lambda1, Huang 2018 alignment-only) ...
run --config student_sub8 --name sub8-sctc-lambda1 \
    --manifest "$SCTC_MANIFEST" --kd-mode sctc --kd-lambda 1.0

# 6) ... then +CTC fine-tune 20ep from its best ckpt (= sctc-lambda1-ctcft)
SCTC_CKPT=$(best_ckpt sub8-sctc-lambda1)
[[ -n "$SCTC_CKPT" ]] || { echo "no ckpt from sub8-sctc-lambda1" >&2; exit 1; }
echo "[suite] sctc ctcft init from: $SCTC_CKPT"
run --config student_sub8 --name sub8-sctc-ctcft \
    --manifest data/train_clean_100.json --kd-mode none \
    --epochs 20 --init-ckpt "$SCTC_CKPT"

# 7) vanilla logit KD (= kd-vanilla-w10)
if [[ -z "${SKIP_VANILLA:-}" ]]; then
  run --config student_sub8 --name sub8-vanilla-w10 \
      --manifest "$FRAME_MANIFEST" --kd-mode logit --kd-weight 10 \
      --temperature 2.0 --blank-mode none
fi

# extras: guided CTC (Kurata 2019) + CR-CTC compute-fair 50ep
if [[ -n "${RUN_EXTRA:-}" ]]; then
  run --config student_sub8 --name sub8-guided-w1 \
      --manifest "$FRAME_T1_MANIFEST" --kd-mode guided --kd-weight 1.0
  run --config student_sub8 --name sub8-cr-fair50 \
      --manifest data/train_clean_100.json --kd-mode cr_ctc \
      --cr-ctc-weight 0.2 --cr-ctc-time-factor 1.5 --epochs 50
fi

echo ""
echo "All sub8 suite runs complete. Evals in analysis/eval_sub8-*.txt"
