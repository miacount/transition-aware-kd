#!/usr/bin/env bash

best_ckpt() {
  local name="$1"
  find "nemo_experiments/$name" -type f -name '*.ckpt' ! -name '*last.ckpt' -print 2>/dev/null \
    | while read -r file; do
        local wer
        wer=$(basename "$file" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
        # A run directory may also contain averaged or manually named
        # checkpoints. Skipping those must not fail under `set -o pipefail`.
        if [[ -n "$wer" ]]; then
          printf '%s %s\n' "$wer" "$file"
        fi
      done \
    | sort -n \
    | head -1 \
    | cut -d' ' -f2-
}

experiment_complete() {
  local name="$1" epochs="$2"
  find "nemo_experiments/$name" -type f -name "*-epoch=${epochs}-last.ckpt" -print -quit 2>/dev/null \
    | grep -q .
}

latest_last_ckpt() {
  local name="$1"
  [[ -d "nemo_experiments/$name" ]] || return 0
  find "nemo_experiments/$name" -type f -name '*last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

run_paper_train() {
  local name="$1" epochs="$2"
  shift 2
  if [[ "${FORCE:-0}" != "1" ]] && experiment_complete "$name" "$epochs"; then
    echo "=== [reuse] complete experiment: $name (${epochs} epochs) ==="
    return 0
  fi

  local resume_ckpt="" resume_version=""
  if [[ "${FORCE:-0}" != "1" ]]; then
    resume_ckpt=$(latest_last_ckpt "$name")
  fi
  if [[ -n "$resume_ckpt" ]]; then
    resume_version=$(basename "$(dirname "$(dirname "$resume_ckpt")")")
    echo "=== [resume] $name (${epochs} epochs) from $resume_ckpt ==="
    bash experiments/train.sh --name "$name" --epochs "$epochs" --seed 1 \
      --resume-version "$resume_version" "$@"
  else
    echo "=== [train] $name (${epochs} epochs) ==="
    bash experiments/train.sh --name "$name" --epochs "$epochs" --seed 1 "$@"
  fi
}

require_file() {
  [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 1; }
}
