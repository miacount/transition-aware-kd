#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: bash experiments/train.sh --name NAME --manifest PATH --kd-mode none|trans|logit [options]

options:
  --kd-weight W        default: 0.0
  --temperature T      default: 2.0
  --epochs N           default: 100
  --lr LR              default: config value
  --subsampling N      default: config value
USAGE
}

NAME=""; MANIFEST=""; KD_MODE="none"; KD_WEIGHT="0.0"; TEMP="2.0"; EPOCHS="100"; LR=""; SUBSAMPLING=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --kd-mode) KD_MODE="$2"; shift 2 ;;
    --kd-weight) KD_WEIGHT="$2"; shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --subsampling) SUBSAMPLING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$NAME" && -n "$MANIFEST" ]] || { usage; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

ARGS=(
  "model.train_ds.manifest_filepath=$MANIFEST"
  "model.kd_mode=$KD_MODE"
  "model.kd_weight=$KD_WEIGHT"
  "model.logit_kd_temperature=$TEMP"
  "trainer.max_epochs=$EPOCHS"
  "exp_manager.name=$NAME"
  "exp_manager.wandb_logger_kwargs.name=$NAME"
)
[[ -n "$LR" ]] && ARGS+=("model.optim.lr=$LR")
[[ -n "$SUBSAMPLING" ]] && ARGS+=("model.encoder.subsampling_factor=$SUBSAMPLING")

python scripts/train.py "${ARGS[@]}"
