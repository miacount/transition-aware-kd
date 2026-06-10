# Transition-aware KD for CTC ASR

Clean experimental pipeline for LibriSpeech CTC student distillation.

## Current Scope

Teacher:

```text
stt_en_conformer_ctc_small
1024 BPE + blank, 4x subsampling
```

Student:

```text
Conformer CTC
1024 BPE + blank
d_model=144, layers=8, heads=4, 4x subsampling
```

Official evaluation splits:

```text
dev_clean, dev_other, test_clean, test_other
```

Official WER is corpus-level WER from `scripts/evaluate_student.py` or epoch-level `val/wer` after the cleanup.

## Preserved Data

Expected manifests:

```text
data/train_clean_100.json
data/dev_clean.json
data/dev_other.json
data/test_clean.json
data/test_other.json
```

Download/rebuild eval manifests:

```bash
python scripts/prepare_librispeech_eval.py \
  --splits dev-clean dev-other test-clean test-other
```

## Experiments

All experiment entrypoints live under `experiments/`. Common runners accept args; presets only call runners.

No-KD baseline:

```bash
bash experiments/presets/00_no_kd.sh
```

Transition KD:

```bash
bash experiments/presets/10_build_transition_targets.sh
bash experiments/presets/11_transition_kd_w025.sh
```

Vanilla frame logit KD:

```bash
bash experiments/presets/20_build_frame_topk_targets.sh
bash experiments/presets/21_vanilla_logit_kd_w01.sh
```

Generic training:

```bash
bash experiments/train.sh \
  --name my-run \
  --manifest data/train_clean_100.json \
  --kd-mode none
```

## Evaluation

Find checkpoints:

```bash
find nemo_experiments -type f -name "*.ckpt"
```

Evaluate a checkpoint:

```bash
bash experiments/eval.sh --ckpt path/to/checkpoint.ckpt --name my-run
```

CTC path mismatch diagnostic:

```bash
bash experiments/diagnose.sh --ckpt path/to/checkpoint.ckpt --name my-run --limit 256
```

## Logging Policy

W&B/TensorBoard should be interpreted through these metrics only:

```text
train/loss
train/ctc_loss
train/kd_loss
val/loss
val/wer
val_wer    # checkpoint monitor alias
```

Prediction sample logging is disabled in the base config.
