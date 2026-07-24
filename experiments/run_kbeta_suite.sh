#!/usr/bin/env bash
# K x beta content/emission decomposition suite for Span-KD.
#
#   L_span = KL(q || s_bar)  +  beta * (-log m_u)
#            \___content___/     \___emission___/
#
# beta=1 reproduces the current loss EXACTLY (unit-tested). This suite isolates
# how much of Span-KD's gain is content (dark knowledge) vs emission (span mass),
# and whether the frozen beta=1 is optimal.
#
# VALIDITY SETUP (why each piece is here):
#  - unit test first: proves the refactor didn't change the beta=1 baseline.
#  - fixed --seed 1, identical kd_weight/epochs/data across cells: deltas reflect
#    the knob, not noise. Single-run noise ~±0.2-0.3 WER.
#  - the beta=1 cell doubles as a REPRODUCTION CONTROL: it must land near
#    12.67/30.64 (test clean/other). If it doesn't, STOP and debug before trusting
#    any delta.
#  - TRIAGE: {beta=1, beta=0} run first. Decision rule:
#      |Δclean| > ~0.5  -> effect exceeds noise: fill grid (1 seed), then 3-seed
#                          the winner + beta=1 anchor for the paper number.
#      |Δclean| <= 0.3  -> within noise: 3-seed the endpoints before concluding.
#  - diagnostics auto-logged to wandb: train/span_{content,emit,m_mean,gate_pass_frac}.
#    Expect monotonic beta↑ => m_mean↑, emit↓. Monotonicity = mechanism validation.
#
# Runs SEQUENTIALLY (single GPU). Launch inside your own tmux.
set -euo pipefail
cd "$(dirname "$0")/.."

W=20
K8_MANIFEST="data/train_clean_100.teacher_only_d6.json"   # existing top-k=8 targets
RAW_MANIFEST="data/train_clean_100.json"

run() { echo -e "\n===== $1 =====\n"; shift; bash experiments/train.sh "$@"; }

# ---- 0. validity gate: beta=1 must equal the original loss numerically --------
echo "===== unit test: content/emission decomposition ====="
python scripts/test_span_beta.py || { echo "UNIT TEST FAILED — aborting"; exit 1; }

# ---- A. beta triage (K=8, reuse existing targets) ----------------------------
run "A1  K8 beta=1  (reproduction control ~12.67/30.64)" \
    --name span_b1_s1 --manifest "$K8_MANIFEST" --kd-mode span_kd --kd-weight $W \
    --span-emit-beta 1.0 --seed 1
run "A2  K8 beta=0  (content-only; predicted large regression)" \
    --name span_b0_s1 --manifest "$K8_MANIFEST" --kd-mode span_kd --kd-weight $W \
    --span-emit-beta 0.0 --seed 1

# ---- A(cont). fill the grid (comment out until triage says the effect is real)-
run "A3  K8 beta=0.25" --name span_b025_s1 --manifest "$K8_MANIFEST" --kd-mode span_kd --kd-weight $W --span-emit-beta 0.25 --seed 1
run "A4  K8 beta=0.5"  --name span_b05_s1  --manifest "$K8_MANIFEST" --kd-mode span_kd --kd-weight $W --span-emit-beta 0.5  --seed 1
run "A5  K8 beta=2"    --name span_b2_s1   --manifest "$K8_MANIFEST" --kd-mode span_kd --kd-weight $W --span-emit-beta 2.0  --seed 1

# ---- B. uniform-target cell (K=8, beta=1): confound-free dark-knowledge test --
# same top-k SET + same emission, teacher relative weights flattened. Compare to
# A1: if A1 beats this, the relative confusability weights (fine dark knowledge)
# matter — without the K-count confound of the K-sweep.
run "B1  K8 beta=1 uniform-target" \
    --name span_uniform_s1 --manifest "$K8_MANIFEST" --kd-mode span_kd --kd-weight $W \
    --span-emit-beta 1.0 --span-uniform-target --seed 1

# ---- C. K sweep (beta=1): rebuild targets for K=1, K=4 (K=8 already exists) ---
for K in 1 4; do
  M="data/train_clean_100.teacher_only_d6_k${K}.json"
  if [[ ! -f "$M" ]]; then
    echo -e "\n===== build K=${K} targets =====\n"
    python scripts/build_span_kd_targets.py \
      --manifest-in "$RAW_MANIFEST" --manifest-out "$M" \
      --out-dir "span_kd_teacher_d6_k${K}" \
      --teacher-only --teacher-blank-penalty 6.0 --where-mode fb --top-k "$K"
  fi
  run "C  K=${K} beta=1" --name "span_k${K}_s1" --manifest "$M" \
      --kd-mode span_kd --kd-weight $W --span-emit-beta 1.0 --seed 1
done
# NOTE: K=8,beta=1 is A1 (span_b1_s1) — shared anchor, don't re-run.

echo -e "\n===== suite complete. Collect test_clean/test_other from each eval. ====="
