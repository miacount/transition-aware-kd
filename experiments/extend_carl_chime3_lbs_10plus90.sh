#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

BEAM_WIDTH="${BEAM_WIDTH:-16}"

# Keep the original experiment names/directories so Lightning restores the
# complete optimizer, Noam scheduler, global-step, and RNG state from epoch 50.
# Although the directory names contain "e50", this script extends their full
# CARL stage to 90 epochs and writes separately named e90 evaluation reports.
CHIME_CONFIG=student_base_chime3_full
CHIME_MANIFEST=data/chime3_train.adapted_teacher.carl.json
CHIME_CLASSIFIER=data/carl_chime3_adapted/teacher_classifier.pt
CHIME_FEATURE=paper-chime3-carl-feature-e10-wu300-lr05-s1
CHIME_FULL=paper-chime3-carl-full-ctc-e50-wu1500-lr05-a1-g1-l1-s1

LBS_CONFIG=student_base
LBS_MANIFEST=data/train_clean_100.small_teacher.carl.json
LBS_CLASSIFIER=data/carl_lbs_small/teacher_classifier.pt
LBS_FEATURE=paper-lbs-carl-feature-e10-wu1000-s1
LBS_FULL=paper-lbs-carl-full-ctc-e50-wu5000-a1-g1-l1-s1

for path in \
  "$CHIME_MANIFEST" "$CHIME_CLASSIFIER" \
  "$LBS_MANIFEST" "$LBS_CLASSIFIER"; do
  require_file "$path"
done

CHIME_FEATURE_CKPT=$(latest_last_ckpt "$CHIME_FEATURE")
LBS_FEATURE_CKPT=$(latest_last_ckpt "$LBS_FEATURE")
CHIME_FULL_CKPT=$(latest_last_ckpt "$CHIME_FULL")
LBS_FULL_CKPT=$(latest_last_ckpt "$LBS_FULL")

for path in \
  "$CHIME_FEATURE_CKPT" "$LBS_FEATURE_CKPT" \
  "$CHIME_FULL_CKPT" "$LBS_FULL_CKPT"; do
  require_file "$path"
  python scripts/check_checkpoint_finite.py "$path"
done

if ! experiment_complete "$CHIME_FULL" 50; then
  echo "CHiME-3 CARL full-stage epoch-50 checkpoint is incomplete" >&2
  exit 1
fi
if ! experiment_complete "$LBS_FULL" 50; then
  echo "LibriSpeech CARL full-stage epoch-50 checkpoint is incomplete" >&2
  exit 1
fi

echo "=== [1/2] CHiME-3 CARL: feature 10 + full 50->90 ==="
export EVAL_MANIFESTS="chime3_dev_real=data/chime3_dev_real_enhanced.json chime3_dev_simu=data/chime3_dev_simu_enhanced.json chime3_eval_real=data/chime3_eval_real_enhanced.json chime3_eval_simu=data/chime3_eval_simu_enhanced.json"
run_paper_train "$CHIME_FULL" 90 \
  --config "$CHIME_CONFIG" --manifest "$CHIME_MANIFEST" --kd-mode carl \
  --carl-classifier "$CHIME_CLASSIFIER" --carl-teacher-dim 176 \
  --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
  --init-ckpt "$CHIME_FEATURE_CKPT" \
  --lr 0.5 --warmup-steps 1500 --save-top-k -1

CHIME_BEST=$(best_ckpt "$CHIME_FULL")
[[ -n "$CHIME_BEST" ]] || { echo "missing CHiME-3 CARL Dev-best checkpoint" >&2; exit 1; }
python scripts/check_checkpoint_finite.py "$CHIME_BEST"
python scripts/evaluate_ctc_decoding.py \
  --ckpt "$CHIME_BEST" --name chime3_carl_e90_devbest \
  --config configs/student_base_chime3_full.yaml --beam-width "$BEAM_WIDTH" \
  --manifest chime3_dev_real=data/chime3_dev_real_enhanced.json \
  --manifest chime3_dev_simu=data/chime3_dev_simu_enhanced.json \
  --manifest chime3_eval_real=data/chime3_eval_real_enhanced.json \
  --manifest chime3_eval_simu=data/chime3_eval_simu_enhanced.json \
  | tee "analysis/beam${BEAM_WIDTH}_chime3_carl_e90_devbest.txt"

echo "=== [2/2] LibriSpeech CARL: feature 10 + full 50->90 ==="
unset EVAL_MANIFESTS
run_paper_train "$LBS_FULL" 90 \
  --config "$LBS_CONFIG" --manifest "$LBS_MANIFEST" --kd-mode carl \
  --carl-classifier "$LBS_CLASSIFIER" --carl-teacher-dim 176 \
  --carl-alpha 1 --carl-gamma 1 --carl-lambda 1 \
  --init-ckpt "$LBS_FEATURE_CKPT" \
  --warmup-steps 5000 --save-top-k -1

LBS_BEST=$(best_ckpt "$LBS_FULL")
[[ -n "$LBS_BEST" ]] || { echo "missing LibriSpeech CARL Dev-best checkpoint" >&2; exit 1; }
python scripts/check_checkpoint_finite.py "$LBS_BEST"
python scripts/evaluate_ctc_decoding.py \
  --ckpt "$LBS_BEST" --name lbs_carl_e90_devbest \
  --config configs/student_base.yaml --beam-width "$BEAM_WIDTH" \
  --manifest dev_clean=data/dev_clean.json \
  --manifest dev_other=data/dev_other.json \
  --manifest test_clean=data/test_clean.json \
  --manifest test_other=data/test_other.json \
  | tee "analysis/beam${BEAM_WIDTH}_lbs_carl_e90_devbest.txt"

echo "[done] CHiME-3 and LibriSpeech CARL 10+90 Dev-best Beam-${BEAM_WIDTH} evaluations"
