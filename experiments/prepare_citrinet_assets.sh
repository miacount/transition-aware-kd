#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "from nemo.collections.asr.models import EncDecCTCModelBPE as M; M.from_pretrained('stt_en_citrinet_512', map_location='cpu'); M.from_pretrained('stt_en_citrinet_256', map_location='cpu')"
teacher_artifact=$(find /root/.cache/torch/NeMo -type f -path '*/stt_en_citrinet_512/*.nemo' -print -quit)
student_artifact=$(find /root/.cache/torch/NeMo -type f -path '*/stt_en_citrinet_256/*.nemo' -print -quit)
[[ -n "$teacher_artifact" && -n "$student_artifact" ]]
mkdir -p tokenizer_citrinet_1024
tar -xf "$teacher_artifact" -C tokenizer_citrinet_1024 ./tokenizer.model ./vocab.txt
teacher_hash=$(tar -xOf "$teacher_artifact" ./tokenizer.model | sha256sum | cut -d' ' -f1)
student_hash=$(tar -xOf "$student_artifact" ./tokenizer.model | sha256sum | cut -d' ' -f1)
local_hash=$(sha256sum tokenizer_citrinet_1024/tokenizer.model | cut -d' ' -f1)
[[ "$teacher_hash" == "$student_hash" && "$teacher_hash" == "$local_hash" ]] || { echo "Citrinet tokenizer mismatch" >&2; exit 1; }
echo "[ok] Citrinet-512/256 tokenizer: $local_hash"
