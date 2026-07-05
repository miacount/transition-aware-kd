#!/usr/bin/env bash
# Token-avg KD with GT-Viterbi forced alignment (official, post-cleanup).
#
# Teacher: GT transcript tokens forced-aligned to teacher posteriors via CTC
# Viterbi. Each GT token's non-blank frame run is one segment; the teacher
# posterior averaged over that segment is distilled onto the student's
# posterior averaged over the same (proportionally scaled) frame interval.
#
# Targets: data/train_clean_100.small_teacher.token_avg.json
#   (build_kd_targets.py --mode token_avg; verified seg_count == GT token count)
# Student: 4x conformer (student_base), same as KD-BE/OCC official runs.
# train.sh runs eval automatically after training.
#
# Usage: bash experiments/run_token_avg_gt_viterbi.sh
#        WEIGHTS="20 15" bash experiments/run_token_avg_gt_viterbi.sh
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.token_avg.json
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

WEIGHTS="${WEIGHTS:-20}"

for w in $WEIGHTS; do
  name="student-token-avg-gt-w${w}-clean"
  echo ""
  echo "════════════════════════════════════════"
  echo "  TRAIN+EVAL  $name  (token_avg, w=$w)"
  echo "════════════════════════════════════════"
  bash experiments/train.sh \
    --name "$name" \
    --manifest "$MANIFEST" \
    --kd-mode token_avg \
    --kd-weight "$w"
done

echo ""
echo "=== Results ==="
echo "--- official baselines (4x student) ---"
grep -E "test_clean|test_other" analysis/eval_student-no-kd.txt 2>/dev/null | sed 's/^/  no-kd: /'
grep -E "test_clean|test_other" analysis/eval_kd-be-occ-best.txt 2>/dev/null | sed 's/^/  kd-be-occ best: /'
echo "--- token-avg GT-Viterbi ---"
for w in $WEIGHTS; do
  f="analysis/eval_student-token-avg-gt-w${w}-clean.txt"
  [[ -f "$f" ]] && grep -E "test_clean|test_other" "$f" | sed "s/^/  w=$w: /"
done
