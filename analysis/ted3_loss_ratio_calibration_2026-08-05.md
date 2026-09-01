# TED3 loss-ratio calibration (2026-08-05)

## Rule

Use the median of the epoch-averaged raw losses over the final five epochs.
For every method, preserve its LibriSpeech weighted auxiliary/CTC ratio on
TED-LIUM3. Test WER is not used to choose a weight.

For lambda mixing, `L = (1 - lambda) * CTC + lambda * KD`.
For additive mixing, `L = CTC + w * KD`.

## Extracted losses and calibrated TED weights

| Method | LS raw CTC | LS raw aux | LS weight | target weighted aux/CTC | TED raw CTC | TED raw aux | calibrated TED weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 77.099373 | 146.665570 | lambda=0.25 | 0.634098 | 44.290310 | 69.144310 | lambda=0.288849 |
| KD-BE | 75.813271 | 90.875916 | lambda=0.25 | 0.399560 | 65.349190 | 33.316490 | lambda=0.439375 |
| Symmetric | 77.234039 | 145.626250 | lambda=0.25, n=4 | 0.628506 | 47.411526 | 66.190666 | lambda=0.310436, n=2 |
| Guided CTC | 78.247360 | -39.388084 | w=1 | 0.503382 (reward magnitude) | 41.148655 | -17.232069 | w=1.202025 |
| Mass3 | 76.545586 | 1.119776 | w=25 | 0.365725 | 42.382477 | 0.937566 | w=16.532378 |
| NTDK | 76.545586 | 1.822872 | w=8 | 0.190516 | 42.382477 | 1.663599 | w=4.853598 |

The TED raw losses come from the completed adapted-teacher runs. They are used
only to translate a pre-existing LS contribution ratio, not to optimize TED
dev/test WER.

## Symmetric n=1 clarification

The completed historical LS `kd-sym-n1-w3` run used additive `kd_weight=3`,
temperature 2, sparse top-8 targets, and the default frame-mean reduction. It
did not use `lambda=1`. Its final-five-epoch weighted KD/CTC ratio was 0.060013.
The `kd-sym-n1-w10` run also used an additive weight, not lambda, and collapsed
to 100% WER. These legacy runs are not used to calibrate the dense-T1,
utterance-sum paper-main comparison because they change several factors at once.

## Exclusions

- S-CTC retains its pure-S-CTC then CTC-finetune schedule.
- CR-CTC is excluded because the available LS reference run collapsed under
  the literal masking recipe. Its corrected TED run must be evaluated separately.
