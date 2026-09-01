#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN=data/tedlium2_train.json
DEV=data/tedlium2_dev.json
TEST=data/tedlium2_test.json

validate_manifests() {
  python - <<'PY'
import json
from pathlib import Path
paths = (
    Path('data/tedlium2_train.json'),
    Path('data/tedlium2_dev.json'),
    Path('data/tedlium2_test.json'),
)
for path in paths:
    rows = [json.loads(line) for line in path.open()]
    assert rows, f"{path}: empty manifest"
    hours = sum(row['duration'] for row in rows) / 3600
    if path.name.endswith('_train.json'):
        assert 180 < hours < 230, (path, hours, 'expected TED2 full train')
    missing = [row['audio_filepath'] for row in rows if not Path(row['audio_filepath']).is_file()]
    assert not missing, f"{path}: {len(missing)} audio files missing; first={missing[:1]}"
    print(f"{path}: {len(rows)} utterances, {hours:.3f}h [OK]")
PY
}

if [[ "${FORCE_DATA:-0}" != "1" && -s "$TRAIN" && -s "$DEV" && -s "$TEST" ]]; then
  validate_manifests
  echo "=== [reuse] TED-LIUM2 full manifests already exist ==="
  exit 0
fi

python scripts/prepare_tedlium.py \
  --root data/TEDLIUM_release2 \
  --out-dir data/tedlium2_full \
  --manifest-prefix data/tedlium2 \
  --text-normalization kaldi \
  --train-hours 0 \
  --min-dur 0.1 \
  --max-dur 30 \
  --keep-all-eval \
  --workers "${TED_PREP_WORKERS:-16}"

validate_manifests
