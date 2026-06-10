#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bash experiments/diagnose.sh --ckpt PATH [--name NAME] [--limit N]" >&2
}

CKPT=""; NAME=""; LIMIT=256
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done
[[ -n "$CKPT" ]] || { usage; exit 1; }
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
mkdir -p analysis
if [[ -z "$NAME" ]]; then NAME="$(basename "$CKPT" .ckpt)"; fi

python scripts/diagnose_ctc_mismatch.py \
  --ckpt "$CKPT" \
  --manifest data/dev_clean.json \
  --limit "$LIMIT" \
  --csv_out "analysis/ctc_mismatch_${NAME}.csv" \
  | tee "analysis/ctc_mismatch_${NAME}.txt"
