#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN=data/tedlium3_train.json
DEV=data/tedlium3_dev.json
TEST=data/tedlium3_test.json

validate_manifests() {
  python - <<'PY'
import json
from pathlib import Path
expected = {
    Path('data/tedlium3_train.json'): (268263, 453.81294444444455),
    Path('data/tedlium3_dev.json'): (507, 1.598175),
    Path('data/tedlium3_test.json'): (1155, 2.617122666666667),
}
for path, (expected_count, expected_hours) in expected.items():
    rows = [json.loads(line) for line in path.open()]
    hours = sum(row['duration'] for row in rows) / 3600
    assert len(rows) == expected_count, (path, len(rows), expected_count)
    assert abs(hours - expected_hours) < 1e-3, (path, hours, expected_hours)
    missing = [row['audio_filepath'] for row in rows if not Path(row['audio_filepath']).is_file()]
    assert not missing, f"{path}: {len(missing)} audio files missing; first={missing[:1]}"
    print(f"{path}: {len(rows)} utterances, {hours:.3f}h [OK]")
PY
}

if [[ "${FORCE_DATA:-0}" != "1" && -s "$TRAIN" && -s "$DEV" && -s "$TEST" ]]; then
  validate_manifests
  echo "=== [reuse] TED-LIUM3 full manifests already exist ==="
  exit 0
fi

python scripts/prepare_tedlium.py \
  --root data/TEDLIUM_release-3 \
  --out-dir data/tedlium3_full \
  --manifest-prefix data/tedlium3 \
  --text-normalization kaldi \
  --train-hours 0 \
  --min-dur 0.1 \
  --max-dur 30 \
  --keep-all-eval \
  --workers "${TED_PREP_WORKERS:-16}"

validate_manifests
