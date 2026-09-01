#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

PLAIN=data/tedlium2_train.json
DENSE=data/tedlium2_train.adapted_teacher.frame_dense_t1.json
SCTC=data/tedlium2_train.adapted_teacher.sctc.json

require_file "$PLAIN"
require_file "$DENSE"
export EVAL_MANIFESTS="ted_dev=data/tedlium2_dev.json ted_test=data/tedlium2_test.json"

# Frozen-LS protocol: transfer all final LS hyperparameters without TED tuning.
run_paper_train paper-ted2-lsfrozen-vanilla-full-l025-s1 100 \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode none --kd-lambda 0.25

run_paper_train paper-ted2-lsfrozen-kdbe-full-l025-s1 100 \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode elimination --kd-lambda 0.25

run_paper_train paper-ted2-lsfrozen-symmetric-full-l025-n4-s1 100 \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode logit \
  --temperature 1 --logit-reduction utterance_sum --blank-mode symmetric \
  --blank-n 4 --kd-lambda 0.25

# Kurata & Audhkhasi: LS-final setting L_CTC + 1 * L_G.
run_paper_train paper-ted2-guided-exact-w1-s1 100 \
  --config student_base_ted2 --manifest "$DENSE" --kd-mode guided \
  --temperature 1 --kd-weight 1

# Huang et al. S-CTC: LS-final compute-matched schedule, 80 + 20 epochs.
if [[ ! -f "$SCTC" ]]; then
  bash experiments/prepare_ted2_sctc_targets.sh
fi
require_file "$SCTC"
run_paper_train paper-ted2-sctc-l1-s1 80 \
  --config student_base_ted2 --manifest "$SCTC" --kd-mode sctc --kd-lambda 1
SCTC_CKPT=$(best_ckpt paper-ted2-sctc-l1-s1)
[[ -n "$SCTC_CKPT" ]] || { echo "no TED2 S-CTC checkpoint found" >&2; exit 1; }
# Fine-tuning is a continuation, not a fresh high-LR training run. The old
# lr=2/warmup=10000 restart destroyed the distilled checkpoint and selected
# FT epoch 0. Use a short warmup and 0.1x Noam scale for the 20-epoch FT phase.
run_paper_train paper-ted2-sctc-l1-ctcft-lr0p2-w1k-s1 20 \
  --config student_base_ted2 --manifest "$PLAIN" --kd-mode none \
  --init-ckpt "$SCTC_CKPT" --lr 0.2 --warmup-steps 1000

# CR-CTC paper protocol: two independent views, alpha=.2, 2.5x both time-mask
# count and maximum width, half physical/effective batch, and half epochs.
# Baseline effective batch is 64, so b16 x accumulation2 gives the required 32.
CR_COMMON=(
  --config student_base_ted2 --manifest "$PLAIN" --kd-mode cr_ctc
  --cr-ctc-weight 0.2 --cr-ctc-warm-step 2000
  --cr-ctc-time-masks-scale 2.5 --cr-ctc-time-width-scale 2.5
  --train-batch-size 16 --accumulate-grad-batches 2
)
run_paper_train smoke-ted2-crctc-paper-e5-b16a2-m2p5w2p5-s1 5 "${CR_COMMON[@]}"
SMOKE_CKPT=$(best_ckpt smoke-ted2-crctc-paper-e5-b16a2-m2p5w2p5-s1)
[[ -n "$SMOKE_CKPT" ]] || { echo "LS-frozen CR-CTC smoke produced no finite checkpoint" >&2; exit 1; }
SMOKE_WER=$(basename "$SMOKE_CKPT" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
awk -v wer="$SMOKE_WER" 'BEGIN { exit !(wer >= 0 && wer < 0.95) }' || {
  echo "LS-frozen CR-CTC smoke failed non-collapse gate: val_wer=$SMOKE_WER" >&2
  exit 1
}
run_paper_train paper-ted2-crctc-paper-e50-b16a2-m2p5w2p5-s1 50 "${CR_COMMON[@]}"

if [[ "${RUN_BEAM:-0}" == "1" ]]; then
  bash experiments/evaluate_paper_main.sh ted2
fi
