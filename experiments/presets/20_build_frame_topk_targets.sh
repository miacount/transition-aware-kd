#!/usr/bin/env bash
set -euo pipefail

bash experiments/build_targets.sh \
  --mode frame_topk \
  --in data/train_clean_100.json \
  --out data/train_clean_100.small_teacher.frame_top8_t1.json \
  --teacher stt_en_conformer_ctc_small \
  --batch-size 16 \
  --top-k 8 \
  --temperature 1.0 \
  --out-dir data/frame_kd_small_top8_t1
