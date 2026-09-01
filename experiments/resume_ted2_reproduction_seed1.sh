#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

NAME=paper-ted2-kdbe-full-l09-s1
VERSION=2026-08-07_11-30-01
DENSE=data/tedlium2_train.adapted_teacher.frame_dense_t1.json

if ! experiment_complete "$NAME" 100; then
  bash experiments/train.sh --name "$NAME" --epochs 100 --seed 1 \
    --config student_base_ted2 --manifest "$DENSE" --kd-mode logit \
    --temperature 1 --logit-reduction utterance_sum \
    --blank-mode elimination --kd-lambda 0.9 \
    --resume-version "$VERSION"
fi

exec env RUN_BEAM=0 bash experiments/run_ted2_reproduction_seed1.sh
