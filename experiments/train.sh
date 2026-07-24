#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: bash experiments/train.sh --name NAME --manifest PATH --kd-mode MODE [options]

kd-mode: none | trans | logit | token_avg | combined | guided | delayed_logit | self_kd | cr_ctc | span_kd | sctc | boundary_kd

options:
  --kd-weight W           KD loss weight (legacy formula, default: 0.0)
  --kd-lambda L           paper formula L=(1-λ)L_CTC+λL_KD; λ=1.0 → pure KD
  --token-avg-weight W    token_avg KD weight for combined mode (default: 0.0)
  --temperature T         logit KD temperature (default: 2.0)
  --blank-mode MODE       none|elimination|symmetric|trim|threshold|random (default: none)
  --blank-n N             symmetric: frames each side of non-blank (default: 1)
  --blank-threshold T     threshold mode: p(blank) < T to include frame (default: 0.9)
  --blank-beta B          random mode: fraction of blank frames to include (default: 0.5)
  --tab-size N            delayed_logit: TAB window size (default: 2)
  --skd-layer N           self_kd: encoder layer for intermediate CTC head (default: 4)
  --skd-alpha A           self_kd: (1-α)*CTC + α*(iCTC+SKD) (default: 0.5)
  --cr-ctc-weight A       cr_ctc: consistency loss weight alpha (default: 0.2)
  --cr-ctc-time-factor F  cr_ctc: time-masking regions/width multiplier vs base SpecAugment (default: 2.5)
  --cr-ctc-warm-step N    cr_ctc: steps to linearly ramp cr loss weight 0 -> alpha (default: 2000)
  --kd-start-step N       CTC-only warm-up steps before KD loss activates (default: 0)
  --span-dual             span_kd: dual-occupancy — pool student with its own de-peaked gamma
  --cross-pool            logit KD: pool both teacher frames per student frame (cross frame-rate)
  --span-student-delta D  span_kd dual: blank penalty for student-side gamma (default: 6.0)
  --span-emit-beta B      span_kd: emission-term weight; 1.0=current loss, 0=content-only (default: 1.0)
  --span-uniform-target   span_kd: flatten teacher top-k to uniform (isolate relative-weight dark knowledge)
  --span-support-mode M   span_kd: occupancy shape ablation — gamma | uniform | rect (default: gamma)
  --span-support-eps E    span_kd --span-support-mode uniform: support threshold, fraction of peak (default: 0.01)
  --span-support-width W  span_kd --span-support-mode rect: window width in frames; 0=per-token n_eff (default: 0)
  --boundary-cell N       boundary_kd: 1 soft | 2 hard proposal | 3 hard m_zero | 4 hard uniform (default: 2)
  --boundary-train-min E  boundary_kd: runtime residual mask threshold (default: 0.0)
  --boundary-weight-src S boundary_kd: residual | gamma_delta (default: residual)
  --span-cnw              span_kd: confidence-need weighting (teacher-decisive & student-weak spans get more KD)
  --span-cnw-alpha A      span_kd cnw: teacher-decisiveness exponent (default: 1.0)
  --span-cnw-gamma G      span_kd cnw: student-need exponent (default: 1.0)
  --init-ckpt PATH        initialize model weights from this .ckpt file (optimizer state discarded)
  --epochs N              (default: 100)
  --lr LR                 (default: config value)
  --subsampling N         (default: config value)
  --config NAME           config name under configs/ (default: student_base)
USAGE
}

NAME=""; MANIFEST=""; KD_MODE="none"; KD_WEIGHT="0.0"; TOKEN_AVG_WEIGHT="0.0"; TEMP="2.0"
EPOCHS="100"; LR=""; SUBSAMPLING=""; CONFIG="student_base"
BLANK_MODE="none"; BLANK_N="1"; BLANK_THRESHOLD="0.9"; BLANK_BETA="0.5"
TAB_SIZE="2"; SKD_LAYER="4"; SKD_ALPHA="0.5"; KD_START_STEP="0"; INIT_CKPT=""
SPAN_DUAL="false"; SPAN_STUDENT_DELTA="6.0"; CROSS_POOL="false"
SPAN_EMIT_BETA="1.0"; SPAN_UNIFORM_TARGET="false"; SEED="1"
SPAN_SUPPORT_MODE="gamma"; SPAN_SUPPORT_EPS="0.01"; SPAN_SUPPORT_WIDTH="0"
BOUNDARY_CELL="2"; BOUNDARY_TRAIN_MIN="0.0"; BOUNDARY_WEIGHT_SRC="residual"
SPAN_CNW="false"; SPAN_CNW_ALPHA="1.0"; SPAN_CNW_GAMMA="1.0"
TRANS_KD_WEIGHT="0.0"; KD_LAMBDA=""; OCC_WEIGHT="0.0"; RES_WEIGHT="0.0"
CR_CTC_WEIGHT="0.2"; CR_CTC_TIME_FACTOR="2.5"; CR_CTC_WARM_STEP="2000"; SR_CTC_WEIGHT="0.0"
LABEL_PRIOR_CTC_ALPHA="0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --kd-mode) KD_MODE="$2"; shift 2 ;;
    --kd-weight) KD_WEIGHT="$2"; shift 2 ;;
    --kd-lambda) KD_LAMBDA="$2"; shift 2 ;;
    --token-avg-weight) TOKEN_AVG_WEIGHT="$2"; shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --blank-mode) BLANK_MODE="$2"; shift 2 ;;
    --blank-n) BLANK_N="$2"; shift 2 ;;
    --blank-threshold) BLANK_THRESHOLD="$2"; shift 2 ;;
    --blank-beta) BLANK_BETA="$2"; shift 2 ;;
    --tab-size) TAB_SIZE="$2"; shift 2 ;;
    --skd-layer) SKD_LAYER="$2"; shift 2 ;;
    --skd-alpha) SKD_ALPHA="$2"; shift 2 ;;
    --kd-start-step) KD_START_STEP="$2"; shift 2 ;;
    --span-dual) SPAN_DUAL="true"; shift 1 ;;
    --cross-pool) CROSS_POOL="true"; shift 1 ;;
    --span-student-delta) SPAN_STUDENT_DELTA="$2"; shift 2 ;;
    --span-emit-beta) SPAN_EMIT_BETA="$2"; shift 2 ;;
    --span-uniform-target) SPAN_UNIFORM_TARGET="true"; shift 1 ;;
    --span-support-mode) SPAN_SUPPORT_MODE="$2"; shift 2 ;;
    --span-support-eps) SPAN_SUPPORT_EPS="$2"; shift 2 ;;
    --span-support-width) SPAN_SUPPORT_WIDTH="$2"; shift 2 ;;
    --boundary-cell) BOUNDARY_CELL="$2"; shift 2 ;;
    --boundary-train-min) BOUNDARY_TRAIN_MIN="$2"; shift 2 ;;
    --boundary-weight-src) BOUNDARY_WEIGHT_SRC="$2"; shift 2 ;;
    --span-cnw) SPAN_CNW="true"; shift 1 ;;
    --span-cnw-alpha) SPAN_CNW_ALPHA="$2"; shift 2 ;;
    --span-cnw-gamma) SPAN_CNW_GAMMA="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --init-ckpt) INIT_CKPT="$2"; shift 2 ;;
    --trans-kd-weight) TRANS_KD_WEIGHT="$2"; shift 2 ;;
    --cr-ctc-weight) CR_CTC_WEIGHT="$2"; shift 2 ;;
    --cr-ctc-time-factor) CR_CTC_TIME_FACTOR="$2"; shift 2 ;;
    --cr-ctc-warm-step) CR_CTC_WARM_STEP="$2"; shift 2 ;;
    --sr-ctc-weight) SR_CTC_WEIGHT="$2"; shift 2 ;;
    --label-prior-ctc-alpha) LABEL_PRIOR_CTC_ALPHA="$2"; shift 2 ;;
    --occ-weight) OCC_WEIGHT="$2"; shift 2 ;;
    --res-weight) RES_WEIGHT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --subsampling) SUBSAMPLING="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$NAME" ]] || { usage; exit 1; }
# self_kd doesn't need an external manifest check — uses plain ASR data
if [[ "$KD_MODE" != "self_kd" ]]; then
  [[ -n "$MANIFEST" ]] || { usage; exit 1; }
  [[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
fi
[[ -z "$MANIFEST" ]] && MANIFEST="data/train_clean_100.json"

ARGS=(
  "model.train_ds.manifest_filepath=$MANIFEST"
  "model.kd_mode=$KD_MODE"
  "model.kd_weight=$KD_WEIGHT"
  "model.token_avg_kd_weight=$TOKEN_AVG_WEIGHT"
  "model.logit_kd_temperature=$TEMP"
  "model.logit_kd_blank_mode=$BLANK_MODE"
  "model.logit_kd_blank_n=$BLANK_N"
  "model.logit_kd_blank_threshold=$BLANK_THRESHOLD"
  "model.logit_kd_blank_beta=$BLANK_BETA"
  "model.logit_kd_tab_size=$TAB_SIZE"
  "model.skd_split_layer=$SKD_LAYER"
  "model.skd_alpha=$SKD_ALPHA"
  "model.kd_start_step=$KD_START_STEP"
  "model.span_kd_dual=$SPAN_DUAL"
  "model.logit_kd_cross_pool=$CROSS_POOL"
  "model.span_kd_student_delta=$SPAN_STUDENT_DELTA"
  "model.span_kd_emit_beta=$SPAN_EMIT_BETA"
  "model.span_kd_uniform_target=$SPAN_UNIFORM_TARGET"
  "model.span_kd_support_mode=$SPAN_SUPPORT_MODE"
  "model.span_kd_support_eps=$SPAN_SUPPORT_EPS"
  "model.span_kd_support_width=$SPAN_SUPPORT_WIDTH"
  "model.boundary_kd_cell=$BOUNDARY_CELL"
  "model.boundary_kd_train_min=$BOUNDARY_TRAIN_MIN"
  "model.boundary_kd_weight_source=$BOUNDARY_WEIGHT_SRC"
  "model.span_kd_cnw=$SPAN_CNW"
  "model.span_kd_cnw_alpha=$SPAN_CNW_ALPHA"
  "model.span_kd_cnw_gamma=$SPAN_CNW_GAMMA"
  "model.seed=$SEED"
  "model.transition_kd_weight=$TRANS_KD_WEIGHT"
  "model.logit_kd_occ_weight=$OCC_WEIGHT"
  "model.logit_kd_res_weight=$RES_WEIGHT"
  "model.cr_ctc_weight=$CR_CTC_WEIGHT"
  "model.cr_ctc_time_mask_factor=$CR_CTC_TIME_FACTOR"
  "model.cr_ctc_warm_step=$CR_CTC_WARM_STEP"
  "model.sr_ctc_weight=$SR_CTC_WEIGHT"
  "model.label_prior_ctc_alpha=$LABEL_PRIOR_CTC_ALPHA"
  "trainer.max_epochs=$EPOCHS"
)
[[ -n "$KD_LAMBDA" ]] && ARGS+=("model.kd_lambda=$KD_LAMBDA")
ARGS+=(
  "exp_manager.name=$NAME"
  "exp_manager.wandb_logger_kwargs.name=$NAME"
)
[[ -n "$LR" ]] && ARGS+=("model.optim.lr=$LR")
[[ -n "$SUBSAMPLING" ]] && ARGS+=("model.encoder.subsampling_factor=$SUBSAMPLING")
# '+' appends the key (not in struct config); quotes make Hydra treat '=' in the
# checkpoint path (val_wer=..-epoch=..) as literal.
[[ -n "$INIT_CKPT" ]] && ARGS+=("+init_from_checkpoint='$INIT_CKPT'")

python scripts/train.py --config-name "$CONFIG" "${ARGS[@]}"

echo ""
echo "=== Training done. Finding best checkpoint... ==="
CKPT=$(ls nemo_experiments/"$NAME"/*/checkpoints/*.ckpt 2>/dev/null \
  | grep -v last \
  | while read -r f; do
      wer=$(basename "$f" | sed -n 's/.*val_wer=\([0-9.]*\)-epoch.*/\1/p')
      echo "$wer $f"
    done \
  | sort -n \
  | head -1 \
  | cut -d' ' -f2-)

if [[ -z "$CKPT" ]]; then
  echo "[eval] No checkpoint found under nemo_experiments/$NAME" >&2
  exit 1
fi
echo "[eval] checkpoint: $CKPT"
bash experiments/eval.sh --ckpt "$CKPT" --name "$NAME" --config "configs/${CONFIG}.yaml"
