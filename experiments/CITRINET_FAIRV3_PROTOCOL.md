# Citrinet fair-v3 tuning and comparison protocol

Status: frozen for seed 1. Do not change candidates after inspecting held-out/test results.
Multi-seed execution is intentionally deferred.

## Common controls

- Student/teacher: Citrinet-144 / Citrinet-256.
- Data: LibriSpeech train-clean-100 and the same cached teacher targets.
- Initialization: the same `arch-citrinet144-t256-lbs-cal100-shared-ctc10-s1` weights.
- Optimizer controls: batch 64, gradient accumulation 8, LR 0.025, bf16, seed 1.
- Candidate selection: best `dev_clean` validation WER only.
- Tuning never loads dev-other, test-clean, or test-other.
- Held-out/test beam-16 decoding runs once, after scales and recipes are frozen.

## Equal tuning budget

Each KD method gets exactly three ordered loss-scale candidates (`lo`, `mid`, `hi`).
The center is the current calibrated setting; low/high bracket it by approximately 0.5x/2x.
S-CTC uses log-odds-adjacent lambdas because lambda must remain below 1.

| Method | Low | Mid | High | Candidate training |
|---|---:|---:|---:|---|
| ATDK `(primary, NTDK)` | `(10,3)` | `(20,6)` | `(40,12)` | 30 epochs |
| Vanilla KD lambda | 0.125 | 0.25 | 0.50 | 30 epochs |
| KD-BE lambda | 0.125 | 0.25 | 0.50 | 30 epochs |
| Symmetric KD lambda | 0.125 | 0.25 | 0.50 | 30 epochs |
| Guided weight | 2.5 | 5 | 10 | 30 epochs |
| S-CTC lambda | 0.96 | 0.98 | 0.99 | 25 S-CTC + 5 CTC epochs |
| CR-CTC weight | 0.25 | 0.5 | 1.0 | 15 two-view epochs (=30 forward-epoch equivalents) |
| FPKD BKL/NBF scale | 0.5 | 1 | 2 | shared 5+5 prefix, then 20 epochs/candidate |
| CARL auxiliary scale | 0.5 | 1 | 2 | shared 5-epoch feature prefix, then 25 epochs/candidate |

Sharing deterministic FPKD/CARL prefixes is compute caching, not extra tuning information: every
candidate starts from the identical checkpoint and receives the same candidate-specific horizon.

## Final seed-1 recipes

- No-KD and single-stage KD: shared CTC10 + 120 continuation epochs.
- S-CTC: shared CTC10 + 96 S-CTC + 24 supervised CTC fine-tuning epochs. The 80/20
  method-stage ratio is preserved while the continuation budget grows to 120 epochs.
- CR-CTC: shared CTC10 + 60 two-view epochs. This matches approximately 120 single-view encoder
  forward epochs; its optimizer warmup and CR ramp are halved proportionally.
- FPKD: shared CTC10 + 10 DFKD + 10 FRKD + 100 PKD. Stage warmups are 100/100/1000 rather than
  restarting a 1000-step warmup inside stages shorter than 1000 steps.
- CARL: shared CTC10 + 10 feature + 110 full. Stage warmups are 100/1100.

Every non-CR final branch receives 120 post-warm-start method epochs (130 data epochs including CTC10).
CR-CTC receives 60 physical two-view epochs, equal to 120 single-view encoder-forward epochs.

The staged recipes necessarily reset optimizer state at method-defined boundaries. They are therefore
recipe-faithful and exposure-matched, while CR-CTC is forward-compute-matched. Exact wall-clock FLOP
matching is not claimed and should be reported separately.
