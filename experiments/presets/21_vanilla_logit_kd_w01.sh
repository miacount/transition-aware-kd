#!/usr/bin/env bash
set -euo pipefail

MANIFEST=data/train_clean_100.small_teacher.frame_top8_t1.json
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST; run experiments/presets/20_build_frame_topk_targets.sh first" >&2; exit 1; }

bash experiments/train.sh \
  --name student-logit-top8-t1-w01-clean \
  --manifest "$MANIFEST" \
  --kd-mode logit \
  --kd-weight 0.1
