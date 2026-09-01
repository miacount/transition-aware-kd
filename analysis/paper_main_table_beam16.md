# LibriSpeech final results

WER (%); lower is better. Greedy is reported on all dev/test splits; no-LM CTC beam decoding is reported on test only.
Checkpoints are selected by minimum dev-clean greedy WER. Bold denotes the best observed value in each column.

| Method | dev-clean<br>Greedy | dev-other<br>Greedy | test-clean<br>Greedy | test-other<br>Greedy | test-clean<br>Beam-16 | test-other<br>Beam-16 |
|---|---:|---:|---:|---:|---:|---:|
| No KD | 14.90 | 33.54 | 15.24 | 34.26 | 14.98 | 34.14 |
| Vanilla frame KD <sup>[1]</sup> | 13.10 | 30.24 | 13.23 | 31.41 | 12.67 | 30.76 |
| Blank elimination KD <sup>[2]</sup> | 12.42 | 29.95 | 12.83 | 30.83 | 12.53 | 30.51 |
| Symmetric selection KD <sup>[3]</sup> | 13.09 | 30.71 | 13.47 | 31.65 | 12.89 | 30.89 |
| Guided CTC <sup>[4]</sup> | 13.24 | 31.49 | 13.46 | 31.86 | 13.17 | 31.57 |
| S-CTC + CTC fine-tune <sup>[5]</sup> | 14.34 | 32.36 | 14.54 | 32.98 | 14.43 | 32.79 |
| CR-CTC <sup>[6]</sup> | 13.84 | 30.94 | 14.02 | 31.84 | 13.50 | 31.18 |
| Mass3 + NTDK (ours) <sup>[7]</sup> | **11.92** | **29.44** | **12.34** | **30.14** | **12.05** | **29.73** |

## Reproduction notes

1. **Vanilla frame KD** — Hinton et al., *Distilling the Knowledge in a Neural Network* (2015); the CTC/TED-style loss recipe follows Hilmes et al. (Interspeech 2025).
2. **Blank elimination KD** — Tian et al., *Knowledge Distillation For CTC-based Speech Recognition Via Consistent Acoustic Representation Learning (CARL)* (Interspeech 2022); dataset-specific lambda follows Hilmes et al. (2025).
3. **Symmetric selection KD** — Hilmes et al., *Analyzing the Importance of Blank for CTC-Based Knowledge Distillation* (Interspeech 2025).
4. **Guided CTC** — Kurata and Audhkhasi, *Guiding CTC Posterior Spike Timings for Improved Posterior Fusion and Knowledge Distillation* (Interspeech 2019). Exact selected-posterior objective `-sum p_S`, weight 1.
5. **S-CTC + CTC fine-tune** — Huang et al., *Knowledge Distillation for Sequence Model* (Interspeech 2018). Transcript-constrained teacher FB occupancy followed by supervised CTC fine-tuning; local compute-matched schedule is 80+20 epochs.
6. **CR-CTC** — Yao et al., *Consistency Regularization for CTC-based Speech Recognition* (ICLR 2025). Faithful two-view consistency loss; the table uses the stable local 50-epoch recipe (physical batch 32, time-mask factor 1.5), not the collapsed strict-mask run.
7. **Mass3 + NTDK** — proposed occurrence-aware hierarchical CTC distillation; `L_CTC + 25 L_Mass3 + 8 L_NTDK`.
