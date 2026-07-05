#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bash experiments/eval.sh --ckpt PATH [--name NAME] [--config PATH]" >&2
}

CKPT=""; NAME=""; CONFIG="configs/student_base.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done
[[ -n "$CKPT" ]] || { usage; exit 1; }
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
mkdir -p analysis
if [[ -z "$NAME" ]]; then NAME="$(basename "$CKPT" .ckpt)"; fi

python scripts/evaluate_student.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --manifest dev_clean=data/dev_clean.json \
  --manifest dev_other=data/dev_other.json \
  --manifest test_clean=data/test_clean.json \
  --manifest test_other=data/test_other.json \
  | tee "analysis/eval_${NAME}.txt"
