#!/usr/bin/env bash
# δ=3 NARROW-width probe for Span-KD.
#
# WHY THIS RUN (the decision it settles):
#   The width sweep so far only covered δ>=6:  d6 12.67/30.64  ->  d9 12.86/30.93
#   ->  d12 ~14 (val 14.31). The WIDE side is monotone WORSE (neighbor
#   contamination). The NARROW side (δ<6) was never measured -- METHOD_REPORT only
#   *hypothesizes* "너무 좁으면 frame-KD로 퇴화" (degenerates to frame-KD, re-hitting
#   the P2 alignment problem) without a data point.
#
#   δ=3 is that missing point, and it decides Method-3 (span pyramid):
#     - d3 clearly WORSE than d6  -> δ=6 is a clean interior optimum, BOTH sides
#       worse -> a width pyramid has no room (averaging in worse targets can only
#       hurt). Method-3 is effectively dead; pivot to Method-2 (masked).
#     - d3 ~= d6, or different error profile -> narrow-side pyramid {3,6} has a
#       case; proceed to build it.
#
# Same recipe as SOTA span-kd-teacher-d6-w20 (base student, teacher-only fb,
# top-k=8, kd_weight=20, emit_beta=1, seed 1) -- ONLY --teacher-blank-penalty
# changes 6.0 -> 3.0, so the comparison is a clean single-knob delta.
#
# Runs SEQUENTIALLY on one GPU. Launch inside your own tmux.
set -euo pipefail
cd "$(dirname "$0")/.."

W=20
RAW_MANIFEST="data/train_clean_100.json"
D3_MANIFEST="data/train_clean_100.teacher_only_d3.json"

# ---- 1. build δ=3 targets (teacher inference over train-100; ~28.5k utts) -----
if [[ ! -f "$D3_MANIFEST" ]]; then
  echo -e "\n===== build δ=3 span targets =====\n"
  python scripts/build_span_kd_targets.py \
    --manifest-in "$RAW_MANIFEST" --manifest-out "$D3_MANIFEST" \
    --out-dir "span_kd_teacher_d3" \
    --teacher-only --teacher-blank-penalty 3.0 --where-mode fb --top-k 8
else
  echo "[skip] $D3_MANIFEST already exists"
fi

# ---- 2. train span-KD @ δ=3 (identical to d6-w20 except the width) -----------
echo -e "\n===== train span-d3-w20 (seed 1) =====\n"
bash experiments/train.sh \
  --name span-d3-w20 --manifest "$D3_MANIFEST" \
  --kd-mode span_kd --kd-weight $W --span-emit-beta 1.0 --seed 1

echo -e "\n===== done. compare test_clean/other vs d6 (12.67/30.64), d9 (12.86/30.93) ====="
