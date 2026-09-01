# LibriSpeech dev-clean + DEMAND: clean to +5 dB

All conditions contain the same 2,703 paired utterances and use the same DEMAND
environment, channel, crop, and mixing seed. WER is no-LM beam-16. Values are
mean ± sample SD over three matched training seeds. RERR is calculated within
each seed as `100 * (WER_Vanilla - WER_AT-DKD) / WER_Vanilla` and then averaged.

| SNR (dB) | Vanilla KD WER (%) | AT-DKD WER (%) | Paired RERR (%) |
|---:|---:|---:|---:|
| Clean | 12.62 ± 0.08 | 11.92 ± 0.09 | 5.55 ± 1.22 |
| +15 | 15.79 ± 0.14 | 14.99 ± 0.16 | 5.04 ± 1.65 |
| +10 | 20.69 ± 0.25 | 19.93 ± 0.43 | 3.71 ± 2.26 |
| +5 | 30.27 ± 0.19 | 29.92 ± 0.63 | 1.16 ± 2.01 |

## Seed-level values

| SNR (dB) | Vanilla KD seeds 1/2/3 | AT-DKD seeds 1/2/3 | RERR seeds 1/2/3 (%) |
|---:|---|---|---|
| Clean | 12.7073 / 12.5731 / 12.5804 | 11.8231 / 11.9444 / 11.9885 | 6.958 / 5.000 / 4.705 |
| +15 | 15.9461 / 15.6722 / 15.7494 | 14.8450 / 14.9756 / 15.1557 | 6.905 / 4.445 / 3.770 |
| +10 | 20.8356 / 20.4110 / 20.8375 | 19.5213 / 19.8724 / 20.3834 | 6.308 / 2.639 / 2.179 |
| +5 | 30.3077 / 30.0614 / 30.4364 | 29.2581 / 29.9787 / 30.5136 | 3.463 / 0.275 / -0.254 |
