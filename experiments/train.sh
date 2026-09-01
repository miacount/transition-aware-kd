#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: bash experiments/train.sh --name NAME --manifest PATH --kd-mode MODE [options]

kd-mode: none | trans | logit | token_avg | combined | guided | delayed_logit | self_kd | cr_ctc | span_kd | sctc | boundary_kd | free_emit | frame_dkd | fpkd_dfkd | fpkd_frkd | fpkd_pkd | carl_feature | carl

options:
  --kd-weight W           KD loss weight (legacy formula, default: 0.0)
  --kd-lambda L           paper formula L=(1-λ)L_CTC+λL_KD; λ=1.0 → pure KD
  --token-avg-weight W    token_avg KD weight for combined mode (default: 0.0)
  --temperature T         logit KD temperature (default: 2.0)
  --logit-reduction R     frame_mean|utterance_sum (paper blank-KD uses utterance_sum)
  --blank-mode MODE       none|elimination|symmetric|trim|threshold|random (default: none)
  --blank-n N             symmetric: frames each side of non-blank (default: 1)
  --blank-threshold T     threshold mode: p(blank) < T to include frame (default: 0.9)
  --blank-beta B          random mode: fraction of blank frames to include (default: 0.5)
  --tab-size N            delayed_logit: TAB window size (default: 2)
  --skd-layer N           self_kd: encoder layer for intermediate CTC head (default: 4)
  --skd-alpha A           self_kd: (1-α)*CTC + α*(iCTC+SKD) (default: 0.5)
  --cr-ctc-weight A       cr_ctc: consistency loss weight alpha (default: 0.2)
  --cr-ctc-time-factor F  cr_ctc: time-masking regions/width multiplier vs base SpecAugment (default: 2.5)
  --cr-ctc-time-masks-scale F  cr_ctc: multiply number of time masks
  --cr-ctc-time-width-scale F  cr_ctc: multiply maximum time-mask width
  --cr-ctc-warm-step N    cr_ctc: steps to linearly ramp cr loss weight 0 -> alpha (default: 2000)
  --frame-dkd-alpha A     DKD TCKD weight (paper default: 1)
  --frame-dkd-beta B      DKD NCKD weight (paper default: 8)
  --frame-dkd-temp T      DKD temperature (paper default: 4)
  --frame-dkd-warmup N    DKD warm-up epochs (paper default: 20)
  --fpkd-bkl-weight W    FPKD binary blank KL weight (default: 1)
  --fpkd-nbf-weight W    FPKD nonblank-frame conditional KL weight (default: 1)
  --fpkd-temp T          FPKD posterior temperature (default: 1)
  --carl-classifier PATH frozen teacher classifier artifact from build_carl_targets.py
  --carl-teacher-dim D   teacher hidden dimension (TED2 adapted teacher: 176)
  --carl-temp T          CARL posterior temperature (default: 1)
  --carl-alpha A         original-head BE-KD weight (default: 1)
  --carl-gamma G         teacher-classifier CTC weight (default: 1)
  --carl-lambda L        teacher-classifier BE-KD weight (default: 1)
  --kd-start-step N       CTC-only warm-up steps before KD loss activates (default: 0)
  --span-dual             span_kd: dual-occupancy — pool student with its own de-peaked gamma
  --cross-pool            logit KD: pool both teacher frames per student frame (cross frame-rate)
  --span-student-delta D  span_kd dual: blank penalty for student-side gamma (default: 6.0)
  --span-emit-beta B      span_kd: emission-term weight; 1.0=current loss, 0=content-only (default: 1.0)
  --span-ntdk-weight W    span_kd: direct weight for raw non-target top-M+tail KL (default: 0.0)
  --span-shoulder-weight W span_kd: GT-emission weight on residual [gamma_delta-gamma_0]+ (default: 0.0)
  --span-ntdk-reliability span_kd: reliability-weight NTDK using cached Mass3 targets
  --span-ntdk-mass-weighted span_kd: exact hierarchy; multiply conditional NTDK by teacher NT mass
  --span-primary-mode M   span_kd: legacy | hard_gt | mass3 | gt_nt (default: legacy)
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
  --sub-weight W          free_emit substitution timing-only weight (default: 0.25)
  --spread-weight W       free_emit residual spread weight (default: 0.05)
  --margin-start N        free_emit curriculum starting margin (default: 0)
  --anneal-epochs N       free_emit margin anneal endpoint (default: 70)
  --init-ckpt PATH        initialize model weights from this .ckpt file (optimizer state discarded)
  --epochs N              (default: 100)
  --lr LR                 (default: config value)
  --warmup-steps N       override Noam warmup steps (for low-LR fine-tuning)
  --subsampling N         (default: config value)
  --config NAME           config name under configs/ (default: student_base)
  --train-batch-size N    physical training batch size (default: config value)
  --accumulate-grad-batches N  gradient accumulation steps (default: config value)
  --resume-version VERSION  resume the same NeMo experiment version from its last checkpoint
  --save-top-k N          override checkpoint retention (-1 keeps every epoch)
  --dev-only              tune on validation only; never evaluate dev-other/test splits
USAGE
}

NAME=""; MANIFEST=""; KD_MODE="none"; KD_WEIGHT="0.0"; TOKEN_AVG_WEIGHT="0.0"; TEMP="2.0"
EPOCHS="100"; LR=""; SUBSAMPLING=""; CONFIG="student_base"
LOGIT_REDUCTION="frame_mean"; TRAIN_BATCH_SIZE=""; ACCUMULATE_GRAD_BATCHES=""
RESUME_VERSION=""; SAVE_TOP_K=""
BLANK_MODE="none"; BLANK_N="1"; BLANK_THRESHOLD="0.9"; BLANK_BETA="0.5"
TAB_SIZE="2"; SKD_LAYER="4"; SKD_ALPHA="0.5"; KD_START_STEP="0"; INIT_CKPT=""
SPAN_DUAL="false"; SPAN_STUDENT_DELTA="6.0"; CROSS_POOL="false"
SPAN_EMIT_BETA="1.0"; SPAN_UNIFORM_TARGET="false"; SEED="1"
SPAN_NTDK_WEIGHT="0.0"; SPAN_NTDK_RELIABILITY="false"; SPAN_NTDK_MASS_WEIGHTED="false"
SPAN_SHOULDER_WEIGHT="0.0"
SPAN_PRIMARY_MODE="legacy"
SPAN_SUPPORT_MODE="gamma"; SPAN_SUPPORT_EPS="0.01"; SPAN_SUPPORT_WIDTH="0"
BOUNDARY_CELL="2"; BOUNDARY_TRAIN_MIN="0.0"; BOUNDARY_WEIGHT_SRC="residual"
SPAN_CNW="false"; SPAN_CNW_ALPHA="1.0"; SPAN_CNW_GAMMA="1.0"
FREE_EMIT_SUB_WEIGHT="0.25"; FREE_EMIT_SPREAD_WEIGHT="0.05"; FREE_EMIT_MARGIN_START="0"; FREE_EMIT_ANNEAL_EPOCHS="70"
TRANS_KD_WEIGHT="0.0"; KD_LAMBDA=""; OCC_WEIGHT="0.0"; RES_WEIGHT="0.0"
CR_CTC_WEIGHT="0.2"; CR_CTC_TIME_FACTOR="2.5"; CR_CTC_WARM_STEP="2000"; SR_CTC_WEIGHT="0.0"
CR_CTC_TIME_MASKS_SCALE=""; CR_CTC_TIME_WIDTH_SCALE=""
LABEL_PRIOR_CTC_ALPHA="0.0"
FRAME_DKD_ALPHA="1.0"; FRAME_DKD_BETA="8.0"; FRAME_DKD_TEMP="4.0"; FRAME_DKD_WARMUP="20"
FPKD_BKL_WEIGHT="1.0"; FPKD_NBF_WEIGHT="1.0"; FPKD_TEMP="1.0"
CARL_CLASSIFIER=""; CARL_TEACHER_DIM="176"; CARL_TEMP="1.0"
CARL_ALPHA="1.0"; CARL_GAMMA="1.0"; CARL_LAMBDA="1.0"; WARMUP_STEPS=""
DEV_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --kd-mode) KD_MODE="$2"; shift 2 ;;
    --kd-weight) KD_WEIGHT="$2"; shift 2 ;;
    --kd-lambda) KD_LAMBDA="$2"; shift 2 ;;
    --token-avg-weight) TOKEN_AVG_WEIGHT="$2"; shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --logit-reduction) LOGIT_REDUCTION="$2"; shift 2 ;;
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
    --span-ntdk-weight) SPAN_NTDK_WEIGHT="$2"; shift 2 ;;
    --span-shoulder-weight) SPAN_SHOULDER_WEIGHT="$2"; shift 2 ;;
    --span-ntdk-reliability) SPAN_NTDK_RELIABILITY="true"; shift 1 ;;
    --span-ntdk-mass-weighted) SPAN_NTDK_MASS_WEIGHTED="true"; shift 1 ;;
    --span-primary-mode) SPAN_PRIMARY_MODE="$2"; shift 2 ;;
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
    --sub-weight) FREE_EMIT_SUB_WEIGHT="$2"; shift 2 ;;
    --spread-weight) FREE_EMIT_SPREAD_WEIGHT="$2"; shift 2 ;;
    --margin-start) FREE_EMIT_MARGIN_START="$2"; shift 2 ;;
    --anneal-epochs) FREE_EMIT_ANNEAL_EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --init-ckpt) INIT_CKPT="$2"; shift 2 ;;
    --trans-kd-weight) TRANS_KD_WEIGHT="$2"; shift 2 ;;
    --cr-ctc-weight) CR_CTC_WEIGHT="$2"; shift 2 ;;
    --cr-ctc-time-factor) CR_CTC_TIME_FACTOR="$2"; shift 2 ;;
    --cr-ctc-time-masks-scale) CR_CTC_TIME_MASKS_SCALE="$2"; shift 2 ;;
    --cr-ctc-time-width-scale) CR_CTC_TIME_WIDTH_SCALE="$2"; shift 2 ;;
    --cr-ctc-warm-step) CR_CTC_WARM_STEP="$2"; shift 2 ;;
    --frame-dkd-alpha) FRAME_DKD_ALPHA="$2"; shift 2 ;;
    --frame-dkd-beta) FRAME_DKD_BETA="$2"; shift 2 ;;
    --frame-dkd-temp) FRAME_DKD_TEMP="$2"; shift 2 ;;
    --frame-dkd-warmup) FRAME_DKD_WARMUP="$2"; shift 2 ;;
    --fpkd-bkl-weight) FPKD_BKL_WEIGHT="$2"; shift 2 ;;
    --fpkd-nbf-weight) FPKD_NBF_WEIGHT="$2"; shift 2 ;;
    --fpkd-temp) FPKD_TEMP="$2"; shift 2 ;;
    --carl-classifier) CARL_CLASSIFIER="$2"; shift 2 ;;
    --carl-teacher-dim) CARL_TEACHER_DIM="$2"; shift 2 ;;
    --carl-temp) CARL_TEMP="$2"; shift 2 ;;
    --carl-alpha) CARL_ALPHA="$2"; shift 2 ;;
    --carl-gamma) CARL_GAMMA="$2"; shift 2 ;;
    --carl-lambda) CARL_LAMBDA="$2"; shift 2 ;;
    --sr-ctc-weight) SR_CTC_WEIGHT="$2"; shift 2 ;;
    --label-prior-ctc-alpha) LABEL_PRIOR_CTC_ALPHA="$2"; shift 2 ;;
    --occ-weight) OCC_WEIGHT="$2"; shift 2 ;;
    --res-weight) RES_WEIGHT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --subsampling) SUBSAMPLING="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --accumulate-grad-batches) ACCUMULATE_GRAD_BATCHES="$2"; shift 2 ;;
    --resume-version) RESUME_VERSION="$2"; shift 2 ;;
    --save-top-k) SAVE_TOP_K="$2"; shift 2 ;;
    --dev-only) DEV_ONLY="true"; shift 1 ;;
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
  "model.logit_kd_reduction=$LOGIT_REDUCTION"
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
  "model.span_ntdk_weight=$SPAN_NTDK_WEIGHT"
  "model.span_shoulder_weight=$SPAN_SHOULDER_WEIGHT"
  "model.span_ntdk_reliability=$SPAN_NTDK_RELIABILITY"
  "model.span_ntdk_mass_weighted=$SPAN_NTDK_MASS_WEIGHTED"
  "model.span_primary_mode=$SPAN_PRIMARY_MODE"
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
  "model.free_emit_sub_weight=$FREE_EMIT_SUB_WEIGHT"
  "model.free_emit_spread_weight=$FREE_EMIT_SPREAD_WEIGHT"
  "model.free_emit_margin_start=$FREE_EMIT_MARGIN_START"
  "model.free_emit_anneal_epochs=$FREE_EMIT_ANNEAL_EPOCHS"
  "model.seed=$SEED"
  "model.transition_kd_weight=$TRANS_KD_WEIGHT"
  "model.logit_kd_occ_weight=$OCC_WEIGHT"
  "model.logit_kd_res_weight=$RES_WEIGHT"
  "model.cr_ctc_weight=$CR_CTC_WEIGHT"
  "model.cr_ctc_time_mask_factor=$CR_CTC_TIME_FACTOR"
  "model.cr_ctc_warm_step=$CR_CTC_WARM_STEP"
  "model.sr_ctc_weight=$SR_CTC_WEIGHT"
  "model.label_prior_ctc_alpha=$LABEL_PRIOR_CTC_ALPHA"
  "model.frame_dkd_alpha=$FRAME_DKD_ALPHA"
  "model.frame_dkd_beta=$FRAME_DKD_BETA"
  "model.frame_dkd_temperature=$FRAME_DKD_TEMP"
  "model.frame_dkd_warmup_epochs=$FRAME_DKD_WARMUP"
  "model.fpkd_temperature=$FPKD_TEMP"
  "model.fpkd_bkl_weight=$FPKD_BKL_WEIGHT"
  "model.fpkd_nbf_weight=$FPKD_NBF_WEIGHT"
  "model.carl_teacher_dim=$CARL_TEACHER_DIM"
  "model.carl_temperature=$CARL_TEMP"
  "model.carl_alpha=$CARL_ALPHA"
  "model.carl_gamma=$CARL_GAMMA"
  "model.carl_lambda=$CARL_LAMBDA"
  "trainer.max_epochs=$EPOCHS"
)
[[ -n "$KD_LAMBDA" ]] && ARGS+=("model.kd_lambda=$KD_LAMBDA")
[[ -n "$CR_CTC_TIME_MASKS_SCALE" ]] && ARGS+=("model.cr_ctc_time_masks_scale=$CR_CTC_TIME_MASKS_SCALE")
[[ -n "$CR_CTC_TIME_WIDTH_SCALE" ]] && ARGS+=("model.cr_ctc_time_width_scale=$CR_CTC_TIME_WIDTH_SCALE")
[[ -n "$SAVE_TOP_K" ]] && ARGS+=("exp_manager.checkpoint_callback_params.save_top_k=$SAVE_TOP_K")
[[ -n "$TRAIN_BATCH_SIZE" ]] && ARGS+=("model.train_ds.batch_size=$TRAIN_BATCH_SIZE")
[[ -n "$ACCUMULATE_GRAD_BATCHES" ]] && ARGS+=("trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES")
if [[ -n "$RESUME_VERSION" ]]; then
  ARGS+=(
    "+exp_manager.version=$RESUME_VERSION"
    "exp_manager.resume_if_exists=true"
    "exp_manager.resume_ignore_no_checkpoint=false"
  )
fi
ARGS+=(
  "exp_manager.name=$NAME"
  "exp_manager.wandb_logger_kwargs.name=$NAME"
)
[[ -n "$LR" ]] && ARGS+=("model.optim.lr=$LR")
[[ -n "$WARMUP_STEPS" ]] && ARGS+=("model.optim.sched.warmup_steps=$WARMUP_STEPS")
[[ -n "$CARL_CLASSIFIER" ]] && ARGS+=("model.carl_classifier_path=$CARL_CLASSIFIER")
[[ -n "$SUBSAMPLING" ]] && ARGS+=("model.encoder.subsampling_factor=$SUBSAMPLING")
# '+' appends the key (not in struct config); quotes make Hydra treat '=' in the
# checkpoint path (val_wer=..-epoch=..) as literal.
[[ -n "$INIT_CKPT" ]] && ARGS+=("+init_from_checkpoint='$INIT_CKPT'")
[[ "$DEV_ONLY" == "true" ]] && ARGS+=("~model.test_ds")

python scripts/train.py --config-name "$CONFIG" "${ARGS[@]}"

if [[ "$DEV_ONLY" == "true" ]]; then
  echo ""
  echo "=== Dev-only tuning run complete; held-out/test evaluation intentionally skipped. ==="
  exit 0
fi


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
