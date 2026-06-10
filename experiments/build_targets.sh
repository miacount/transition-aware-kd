#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: bash experiments/build_targets.sh --mode transition|frame_topk --in PATH --out PATH [options]

options:
  --teacher NAME       default: stt_en_conformer_ctc_small
  --batch-size N       default: 16
  --top-k N            default: 8, frame_topk only
  --temperature T      default: 2.0, frame_topk only
  --out-dir PATH       default: data/frame_kd, frame_topk only
  --limit N            default: 0
USAGE
}

MODE=""; IN=""; OUT=""; TEACHER="stt_en_conformer_ctc_small"; BATCH=16; TOPK=8; TEMP=2.0; OUT_DIR="data/frame_kd"; LIMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --in) IN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$MODE" && -n "$IN" && -n "$OUT" ]] || { usage; exit 1; }

python scripts/build_kd_targets.py \
  --mode "$MODE" \
  --manifest_in "$IN" \
  --manifest_out "$OUT" \
  --teacher "$TEACHER" \
  --batch_size "$BATCH" \
  --top_k "$TOPK" \
  --temperature "$TEMP" \
  --out_dir "$OUT_DIR" \
  --limit "$LIMIT"
