# Main-table protocol

## Comparison rule

All rows use the same student architecture, tokenizer, teacher, data split,
optimizer, augmentation, checkpoint selector, and decoder within a dataset.
Published methods are reimplemented as losses in this shared environment; the
numbers are not presented as direct reproductions of each paper's absolute WER,
because those papers use different architectures, units, training corpora, and
language models.

## Datasets

- LibriSpeech: train-clean-100, official dev/test clean/other splits, 100 epochs.
- TED-LIUM3: all 268,263 train supervisions (453.813 h), official legacy dev
  (507 utterances) and test (1,155 utterances), 50 epochs.
- TED text uses Lhotse/Kaldi-style removal of `[NOISE]` and `<unk>` markers,
  clitic joining, and number verbalization required by the fixed alphabetic
  LibriSpeech BPE. No training utterance is dropped because it contains `<unk>`.
- The earlier 164 h data is a clean filtered pilot, not the TED main result.

## Methods and fixed hyperparameters

| Row | LibriSpeech | TED-LIUM3 |
|---|---:|---:|
| No KD | 100 ep | 50 ep |
| Vanilla frame KD | lambda=0.25 | lambda=0.9 |
| Blank elimination KD | lambda=0.25 | lambda=0.9 |
| Symmetric selection KD | lambda=0.25, n=4 | lambda=1, n=2 |
| Guided CTC | exact paper loss, weight=1 | exact paper loss, weight=1 |
| S-CTC + CTC fine-tune | 80 + 20 ep | 40 + 10 ep |
| CR-CTC | stable local run: 50 ep, physical batch 32 | 25 ep, physical batch 16 |
| Mass3 + NTDK (ours) | mass=25, NTDK=8 | same values; no TED tuning |

The Blank-KD values are selected a priori from the published dataset-specific
settings. On LibriSpeech, `lambda=0.25, n=4` belongs to symmetric selection;
blank elimination is reported separately. `full` denotes a full-vocabulary
teacher posterior, not retention of blank frames.

CR-CTC processes two augmented views, so its epochs are halved for a
forward-compute-matched comparison. The LibriSpeech main row uses the validated
non-collapsed local 50-epoch run (`time_mask_factor=1.5`, physical batch 32).
The stricter attempt that multiplied both mask count and width by 2.5 collapsed
and is excluded from the main table rather than reported as a meaningful CR-CTC
result. This distinction must remain explicit when comparing with the paper.

## Selection and evaluation

- Select one checkpoint per run by minimum dev greedy WER.
- Report both greedy CTC and BPE prefix-beam WER with beam width 16 on dev and test.
- Test is not used for checkpoint or hyperparameter selection.
- No external LM is used in the primary table. This isolates acoustic/KD
  transfer and avoids masking dark-knowledge gains. A 4-gram-LM result may be
  reported separately for comparability with papers that decode with an LM.
- Run seeds only after the final method list and protocol are frozen.
