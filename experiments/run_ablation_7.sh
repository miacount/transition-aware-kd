#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paper_suite_common.sh

LS_D6=data/train_clean_100.teacher_only_d6_mass3_ntdk_m32.json
LS_D0=data/train_clean_100.teacher_only_d0_mass3_ntdk_m32.json
LS_PERM=data/train_clean_100.teacher_only_d6_mass3_ntdk_m32_permuted.json
CH_D6=data/chime3_train.adapted_teacher.d6_mass3_ntdk_m32.json
CH_D0=data/chime3_train.adapted_teacher.d0_mass3_ntdk_m32.json
for path in "$LS_D6" "$LS_D0" "$LS_PERM" "$CH_D6" "$CH_D0"; do require_file "$path"; done

run_ablation() {
  local name="$1" config="$2" manifest="$3"; shift 3
  if experiment_complete "$name" 100; then echo "=== [reuse] $name ==="; return 0; fi
  run_paper_train "$name" 100 --config "$config" --manifest "$manifest" --kd-mode span_kd "$@"
}

run_chime_ablation() {
  local name="$1" manifest="$2" mass_weight="$3" ntdk_weight="$4"
  local resume_ckpt="" resume_version=""
  if experiment_complete "$name" 100; then echo "=== [reuse] $name ==="; return 0; fi
  resume_ckpt=$(latest_last_ckpt "$name")
  local resume_args=()
  if [[ -n "$resume_ckpt" ]]; then
    resume_version=$(basename "$(dirname "$(dirname "$resume_ckpt")")")
    resume_args=(
      "+exp_manager.version=$resume_version"
      "exp_manager.resume_if_exists=true"
      "exp_manager.resume_ignore_no_checkpoint=false"
    )
    echo "=== [resume] $name from $resume_ckpt ==="
  else
    echo "=== [train] $name ==="
  fi
  python scripts/train.py --config-name student_base_chime3_runtime \
    "model.train_ds.manifest_filepath=$manifest" \
    model.kd_mode=span_kd model.span_primary_mode=mass3 \
    "model.kd_weight=$mass_weight" "model.span_ntdk_weight=$ntdk_weight" \
    model.optim.lr=0.5 model.optim.sched.warmup_steps=750 model.seed=1 \
    trainer.max_epochs=100 "exp_manager.name=$name" \
    "exp_manager.wandb_logger_kwargs.name=$name" "${resume_args[@]}"
  [[ -n "$(best_ckpt "$name")" ]] || { echo "no finite checkpoint: $name" >&2; exit 1; }
}

run_ablation ablation-lbs-mass3-only-w25-s1 student_base "$LS_D6" \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 0
run_ablation ablation-lbs-ntdk-only-w8-s1 student_base "$LS_D6" \
  --kd-weight 0 --span-primary-mode mass3 --span-ntdk-weight 8
run_ablation ablation-lbs-full-d0-mass25-ntdk8-s1 student_base "$LS_D0" \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8
run_ablation ablation-lbs-full-d6-mass25-permuted-ntdk8-s1 student_base "$LS_PERM" \
  --kd-weight 25 --span-primary-mode mass3 --span-ntdk-weight 8
run_chime_ablation ablation-chime3-mass3-only-w25-lr05-s1 "$CH_D6" 25 0
run_chime_ablation ablation-chime3-ntdk-only-w8-lr05-s1 "$CH_D6" 0 8
run_chime_ablation ablation-chime3-full-d0-mass25-ntdk8-lr05-s1 "$CH_D0" 25 8

echo "=== all seven ablation trainings complete ==="
