#!/usr/bin/env bash
set -euo pipefail

bash experiments/train.sh \
  --name student-none-144x8-1024-clean \
  --manifest data/train_clean_100.json \
  --kd-mode none \
  --kd-weight 0.0
