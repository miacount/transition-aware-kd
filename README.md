# Mass3 + NTDK for CTC ASR

Paper code for occurrence-aware hierarchical knowledge distillation in CTC ASR.
The frozen method uses blank-suppressed forward-backward occupancy for temporal
support and distills raw teacher posteriors with Mass3 and conditional NTDK.

## Frozen method

- Teacher support: blank penalty `delta=6`
- Mass3: blank / ground-truth / non-target mass, weight `25`
- NTDK: conditional non-target distribution, top-M cache `M=32`, weight `8`
- Student: Conformer-CTC, 8 layers, d_model 144, 1024 BPE, 4x subsampling

The detailed method and evidence are in
[`analysis/PAPER_STORYLINE_AND_METHOD.md`](analysis/PAPER_STORYLINE_AND_METHOD.md)
and [`analysis/VALIDATION_FIGURES.md`](analysis/VALIDATION_FIGURES.md).

## Repository layout

- `src/`: training model, data pipeline, CTC forward-backward, KD losses
- `scripts/`: data preparation, target building, training, evaluation, validation
- `experiments/`: reproducible LibriSpeech, TED-LIUM2, and TED-LIUM3 entrypoints
- `configs/`: dataset-specific student and adapted-teacher configs
- `analysis/`: paper tables, final decoding outputs, and validation artifacts
- `figures/`: paper-ready figures
- `data/`, `nemo_experiments/`: local datasets, targets, checkpoints (Git-ignored)

## Reproduction

LibriSpeech-100 and TED-LIUM3 main suite:

```bash
bash experiments/run_paper_main_all.sh
```

TED-LIUM3 with a domain-adapted teacher:

```bash
bash experiments/prepare_ted3_adapted_teacher_targets.sh
bash experiments/run_paper_main_ted3_adapted_teacher.sh
```

TED-LIUM2 pilot (teacher fine-tuning, No-KD, and ours only):

```bash
bash experiments/run_ted2_pilot_all.sh
```

The TED-LIUM manifests use the same Kaldi-style normalization for teacher,
student, and evaluation. STM ignore segments are excluded during preparation.

Evaluate saved checkpoints and rebuild the paper table:

```bash
bash experiments/evaluate_paper_main.sh lbs
bash experiments/evaluate_paper_main.sh ted2
bash experiments/evaluate_paper_main.sh ted3
python scripts/summarize_paper_main.py --beam-width 16
```

## Monitoring

```bash
watch -n 5 'ps -eo etime,cmd | grep -E "build_kd_targets|build_span_kd_targets|scripts/train.py" | grep -v grep'
tail -F nemo_experiments/<run>/<timestamp>/lightning_logs/version_0/metrics.csv
```

TED-LIUM2 pilot runs log to the W&B project
`transition-aware-kd-ted2-pilot`. Checkpoints and generated targets remain local
under `nemo_experiments/` and `data/`; they are intentionally not committed.

## Validation

Protocol tests are under `tests/`. Before committing:

```bash
python -m pytest -q tests
find experiments -name "*.sh" -print0 | xargs -0 -n1 bash -n
```
