#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for path in data/chime3_train_enhanced.json data/chime3_dev_enhanced.json data/chime3_eval_enhanced.json; do
  [[ -f "$path" ]] || { echo "missing required manifest: $path" >&2; exit 1; }
done

python scripts/finetune_teacher_ted3.py --config-name teacher_finetune_chime3
