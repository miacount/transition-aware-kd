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

# Eval manifests default to LibriSpeech; override with EVAL_MANIFESTS
# (space-separated name=path pairs) for other datasets, e.g. TED-LIUM.
if [[ -z "${EVAL_MANIFESTS:-}" ]]; then
  EVAL_MANIFESTS="dev_clean=data/dev_clean.json dev_other=data/dev_other.json \
test_clean=data/test_clean.json test_other=data/test_other.json"
fi
MANIFEST_ARGS=()
for pair in $EVAL_MANIFESTS; do MANIFEST_ARGS+=(--manifest "$pair"); done

python scripts/evaluate_student.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  "${MANIFEST_ARGS[@]}" \
  | tee "analysis/eval_${NAME}.txt"
