#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

NAME=paper-chime3-lr05-mass3-w25-ntdk8-s1
MASS=data/chime3_train.adapted_teacher.d6_mass3_ntdk_m32.json

require_file "$MASS"

# Do not call experiments/train.sh here. Its legacy post-training evaluator
# loads YAML without composing Hydra defaults and therefore rejects the CHiME-3
# runtime config after an otherwise successful training run.
if [[ "${FORCE:-0}" != "1" ]] && experiment_complete "$NAME" 100; then
  echo "=== [reuse] complete experiment: $NAME (100 epochs) ==="
else
  echo "=== [train] ours only: $NAME ==="
  python scripts/train.py --config-name student_base_chime3_runtime \
    "model.train_ds.manifest_filepath=$MASS" \
    model.kd_mode=span_kd \
    model.kd_weight=25 \
    model.span_primary_mode=mass3 \
    model.span_ntdk_weight=8 \
    model.optim.lr=0.5 \
    model.optim.sched.warmup_steps=750 \
    model.seed=1 \
    trainer.max_epochs=100 \
    "exp_manager.name=$NAME" \
    "exp_manager.wandb_logger_kwargs.name=$NAME"
fi

CKPT=$(best_ckpt "$NAME")
[[ -n "$CKPT" ]] || { echo "missing checkpoint for $NAME" >&2; exit 1; }
echo "=== [best] $CKPT ==="

# student_base_ted3.yaml is a fully materialized (non-Hydra-defaults) config
# with the exact same tokenizer and 144x8/subsampling-4 student architecture.
# Dataset paths are replaced explicitly below.
export EVAL_MANIFESTS="chime3_dev_real=data/chime3_dev_real_enhanced.json chime3_dev_simu=data/chime3_dev_simu_enhanced.json chime3_eval_real=data/chime3_eval_real_enhanced.json chime3_eval_simu=data/chime3_eval_simu_enhanced.json"
bash experiments/eval.sh \
  --ckpt "$CKPT" --name "$NAME" --config configs/student_base_ted3.yaml

if [[ "${RUN_BEAM:-1}" == "1" ]]; then
  BEAM_WIDTH="${BEAM_WIDTH:-16}" \
    bash experiments/evaluate_chime3_ours_lr05_fixed.sh
fi
