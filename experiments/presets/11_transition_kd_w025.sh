#!/usr/bin/env bash
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.transition.json
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST; run experiments/presets/10_build_transition_targets.sh first" >&2; exit 1; }

KD_WEIGHT=${1:-0.25}

bash experiments/train.sh \
  --name "student-transition-w${KD_WEIGHT}-clean" \
  --manifest "$MANIFEST" \
  --kd-mode trans \
  --kd-weight "$KD_WEIGHT"
