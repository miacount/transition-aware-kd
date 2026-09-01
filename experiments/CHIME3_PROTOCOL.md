# CHiME-3 teacher adaptation and KD protocol

## Data

- Use only original-speed enhanced waveforms under
  data/chime/CHiME3/data/audio/16kHz/enhanced.
- Train: 8,738 utterances (1,600 real + 7,138 simulated), 18.056 h.
- Dev: 3,280 utterances (1,640 real + 1,640 simulated), 5.640 h.
- Eval: 2,640 utterances (1,320 real + 1,320 simulated), 4.454 h.
- Do not use enhanced_picola variants, numbered annotation shards, BTH, or
  opaque non-WAV feature files.

The manifest builder uses the aggregate annotation dot text, applies the same
alphabetic normalization expected by tokenizer_1024, and verifies every WAV
header and path. Dev selects checkpoints; Eval is report-only.

## Compared systems

1. Adapted teacher: stt_en_conformer_ctc_small fine-tuned for 10 epochs on
   CHiME-3 train, effective batch 64, selected by minimum combined-dev WER.
2. No-KD student: 144x8 Conformer trained from scratch for 100 epochs.
3. Mass3+NTDK student: the identical student and exposure, using targets from
   the selected adapted teacher and fixed hyperparameters weight=25, NTDK=8.

Both students use seed 1, effective batch 64, a 750-step Noam warm-up, the same
SpecAugment, tokenizer, optimizer, dev checkpoint selector, and beam-16 decoder.
Report dev/eval real and simulated WER separately. Do not select anything using
Eval.

## Run

Run the stages separately when monitoring expensive jobs:

    python scripts/prepare_chime3.py
    bash experiments/finetune_teacher_chime3.sh
    bash experiments/prepare_chime3_adapted_teacher_targets.sh
    bash experiments/run_chime3_students.sh

Or run the complete pipeline:

    bash experiments/run_chime3_pipeline.sh

Set RUN_BEAM=0 to skip beam decoding during student training. Run it later with:

    bash experiments/evaluate_chime3.sh

Target generation is resumable. TARGET_BATCH_SIZE can lower its GPU memory use.
