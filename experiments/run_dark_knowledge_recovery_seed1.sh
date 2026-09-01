#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/measure_dark_knowledge_recovery.py \
  --manifest test_clean=data/test_clean.json \
  --manifest test_other=data/test_other.json \
  --teacher stt_en_conformer_ctc_small \
  --config configs/student_base.yaml \
  --delta 6 \
  --top-m 32 \
  --cache-dir analysis/dark_recovery_teacher_cache_d6_m32 \
  --output-json analysis/dark_knowledge_recovery_seed1.json \
  --output-md analysis/dark_knowledge_recovery_seed1.md \
  --model vanilla_frame_kd=nemo_experiments/paper-lbs-vanilla-full-l025-s1/2026-07-29_16-55-41/checkpoints/paper-lbs-vanilla-full-l025-s1--val_wer=0.1301-epoch=93.ckpt \
  --model mass3_only=nemo_experiments/ablation-lbs-mass3-only-w25-s1/2026-08-18_06-58-01/checkpoints/ablation-lbs-mass3-only-w25-s1--val_wer=0.1240-epoch=99.ckpt \
  --model permuted_ntdk=nemo_experiments/ablation-lbs-full-d6-mass25-permuted-ntdk8-s1/2026-08-18_15-08-41/checkpoints/ablation-lbs-full-d6-mass25-permuted-ntdk8-s1--val_wer=0.1206-epoch=98.ckpt \
  --model full_ntdk=nemo_experiments/span-d6-mass3-w25-ntdk-a8-s1/2026-07-28_03-59-48/checkpoints/span-d6-mass3-w25-ntdk-a8-s1--val_wer=0.1190-epoch=98.ckpt
