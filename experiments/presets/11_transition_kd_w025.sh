#!/usr/bin/env bash
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.transition.json
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST; run experiments/presets/10_build_transition_targets.sh first" >&2; exit 1; }

bash experiments/train.sh \
  --name student-transition-w025-clean \
  --manifest "$MANIFEST" \
  --kd-mode trans \
  --kd-weight 0.25
