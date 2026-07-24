#!/usr/bin/env bash
# Time-axis occupancy-shape ablation for Span-KD.
#
#   L_u = -log sum_t g~(t,u) * p_S(t, y_u)
#
# is minimised by p_S = 1 over the WHOLE support of g~, so any g~' with the same
# support has the same argmin — the occupancy SHAPE only steers the optimisation
# path, never the optimum. This suite measures whether the shape (and then the
# width) buys anything, by replacing g~ while holding everything else fixed.
#
# This is the time-axis twin of the vocab-axis uniform-target cell (span_uniform_s1),
# which already showed the vocab axis carries nothing (12.59 vs 12.89).
#
# CELLS (all: student_base, K=8 delta=6 targets, w=20, seed 1, 100ep)
#   T0 gamma      teacher occupancy as-is    -> ANCHOR, already run as span_b1_s1
#                                               (LS test 12.89 / 31.17)
#
# ANCHOR CHOICE (read before comparing to the paper number). The anchor is
# span_b1_s1 (12.89/31.17), NOT the reported span-kd-teacher-d6-w20.0
# (12.67/30.64). The latter ran 2026-07-02 with no model.seed at all (the config
# had no seed key then) and under the pre-refactor pooling code. span_b1_s1 ran
# with --seed 1 on the current code path, which is what these cells use, so it is
# the only control whose delta is attributable to the knob alone. The 0.22 gap
# between the two is most likely run-to-run nondeterminism: at sub4 both code
# changes are provable no-ops (ts == tt makes the 2-point resample equal the old
# nearest lookup, and teacher gamma sums to ~1 per token so the sup_sum > 1e-6
# gate never fires). "Most likely", not verified — if any cell lands within noise
# of the anchor, re-run T0 on the current code before concluding.
#   T1 uniform    same frames (>1% of peak, ~2 on average), flat
#                 -> does the SHAPE matter?
#   T2 rect w=1   the teacher's spike frame only
#                 -> does the span WIDTH matter at all? (the frame-KD floor)
#   T3 rect w=3   3 flat frames centred on the spike, ignoring g's frame set
#                 -> is an arbitrary window as good as g's chosen frames?
#
# VALIDITY
#  - unit test first: mode=gamma must be a bit-exact no-op, so T0 (span_b1_s1)
#    remains a valid anchor and every delta is attributable to the knob.
#  - identical manifest/weight/epochs/seed across cells; targets are NOT rebuilt.
#  - single-run noise ~+/-0.2-0.3 WER: treat |delta| <= 0.3 as a tie and 3-seed
#    the deciding pair before concluding anything for the paper.
#  - de-peak profile is measured per cell: WER can tie while the mechanism differs
#    (e.g. w=1 is expected NOT to de-peak the student). That is a result too.
#
# Runs SEQUENTIALLY on one GPU. Launch inside your own tmux.
set -euo pipefail
cd "$(dirname "$0")/.."

W=20
MANIFEST="data/train_clean_100.teacher_only_d6.json"
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST" >&2; exit 1; }

echo "===== validity gate: span-support-mode unit test ====="
python scripts/test_span_support_mode.py || { echo "UNIT TEST FAILED — aborting"; exit 1; }

best_ckpt() {  # lowest val_wer, excluding -last
  ls nemo_experiments/"$1"/*/checkpoints/*.ckpt 2>/dev/null | grep -v last \
    | while read -r f; do
        wer=$(basename "$f" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
        echo "$wer $f"
      done | sort -n | head -1 | cut -d' ' -f2-
}

run() {
  local name="$1"; shift
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  $(date '+%F %T')  $name"
  echo "════════════════════════════════════════════════════════"
  bash experiments/train.sh --config student_base --name "$name" \
      --manifest "$MANIFEST" --kd-mode span_kd --kd-weight $W --seed 1 "$@"
  local ck; ck=$(best_ckpt "$name")
  [[ -n "$ck" ]] || { echo "no ckpt for $name" >&2; return 1; }
  bash experiments/eval.sh --ckpt "$ck" --name "$name"
}

run span_supp_uniform --span-support-mode uniform --span-support-eps 0.01
run span_supp_rect_w1 --span-support-mode rect --span-support-width 1
run span_supp_rect_w3 --span-support-mode rect --span-support-width 3

# ---- mechanism readout: does the student still de-peak in each cell? ---------
PROFILE_ARGS=(--limit 200)
for n in span_b1_s1 span_supp_uniform span_supp_rect_w1 span_supp_rect_w3; do
  ck=$(best_ckpt "$n") || true
  [[ -n "$ck" ]] && PROFILE_ARGS+=(--ckpt "$n=$ck")
done
python scripts/depeak_student_profile.py "${PROFILE_ARGS[@]}" \
  --out analysis/support_shape_profile.json

echo ""
echo "===== suite complete ====="
echo "WER   : analysis/eval_span_supp_*.txt   (anchor: analysis/eval_span_b1_s1_ls.txt 12.89/31.17)"
echo "profile: analysis/support_shape_profile.json"
