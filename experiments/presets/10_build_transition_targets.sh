#!/usr/bin/env bash
set -euo pipefail

bash experiments/build_targets.sh \
  --mode transition \
  --in data/train_clean_100.json \
  --out data/train_clean_100.small_teacher.transition.json \
  --teacher stt_en_conformer_ctc_small \
  --batch-size 16
