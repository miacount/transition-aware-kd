# Published-baseline audit (2026-07-29)

## Scope

- Common environment: LibriSpeech train-clean-100, Conformer 8x144, BPE 1024, seed 1.
- Proposed method: `L_CTC + 25 L_Mass3 + 8 L_NTDK`, 100 epochs.
- Checkpoint selection: minimum validation WER on dev-clean only.
- Decoding below: CTC prefix beam, beam width 16, no external LM, common pruning settings.

## Reproduction audit

| Local label | Loss-level verdict | Important difference from the paper |
|---|---|---|
| Vanilla frame KD | Adaptation, not exact full-posterior reproduction | Teacher posterior is temperature-scaled with T=2, truncated to top-8, and renormalized. The local loss is `L_CTC + w L_KD`; no T-squared compensation is used. |
| KD-BE | Selection mask is correct; posterior/loss recipe is adapted | Frames whose teacher argmax is blank are removed exactly as in KD-BE. However, it uses the same T=2/top-8 target approximation and additive weight 10 instead of the paper's exact interpolation form. |
| Symmetric blank selection | Selection mask is correct; selected run is not the paper's principal recipe | The n-frame dilation around teacher nonblank spikes is correct. The reported local run is n=1 with `L_CTC + 3 L_KD`; it is not pure KD (`lambda=1`) and uses T=2/top-8 targets. |
| Guided CTC | Not exact | The paper maximizes the selected student posterior (`-sum p_s`) at teacher nonblank argmax positions. Local code minimizes hard cross-entropy (`-sum log p_s`). The mask is correct but the objective is different. |
| S-CTC | Core loss is faithful | Teacher transcript-constrained forward-backward occupancies, including residual blank occupancy, supervise the student frame posterior as in Eq. 10. The 20-epoch CTC fine-tune is also consistent with the paper's reported protocol, but total local training is 100+20 epochs. |
| CR-CTC | Eq. 3-4 loss is faithful; training recipe is not exact | Two independent views, averaged CTC, stop-gradient symmetric frame KL, and alpha=0.2 are correct. Current runs use a weaker/custom time-mask scaling and do not halve batch size. The 100-epoch run uses about twice the forward compute of a 100-epoch single-view model. |

## Weight-search audit

- Vanilla: weights 1 and 10 only.
- KD-BE: only weight 10 for the published method; other BE-named runs add local occupancy/residual components and are not KD-BE baselines.
- Symmetric: lambda 0.5, incomplete lambda 1, and additive weights 3/5/10; n=2 was only paired with weight 10. This is not a balanced grid.
- Guided CTC: weight 1 only.
- S-CTC: pure S-CTC followed by CTC fine-tuning; no interpolation sweep.
- CR-CTC: alpha 0.2 only; several debugging/time-mask variants, but no clean alpha sweep.
- The proposed method received materially more development/tuning than most baselines.

Using the same numeric weight for all methods is not fair because their loss normalizations differ. A defensible protocol is: (1) exact paper/recommended setting, and (2) a separately labeled dev-tuned setting with the same trial budget per method. Hyperparameters and checkpoints must be chosen using dev-clean only; test splits are report-only.

## CR-CTC compute audit

- `cr_ctc_w02_v5`: 100 epochs, physical batch 32, accumulation 2, two forwards per utterance. This is the strong, roughly 2x-forward-compute result.
- `cr_ctc_w02_fair50`: 50 epochs, but physical batch 32 and accumulation 2 were unchanged. This is not the paper's half-batch/half-epoch recipe and has only about half as many optimizer updates as the corresponding 100-epoch baseline.
- A faithful local fair run should use 50 epochs, physical batch 16, accumulation 2, alpha 0.2, and the paper's 2.5x increase to both time-mask count and maximum mask fraction. With the current `volume_factor -> sqrt` implementation, the latter corresponds to `cr_ctc_time_mask_factor=6.25`, not 1.5.

## No-LM beam-16 results

| Method | dev-clean | dev-other | test-clean | test-other | 4-way avg |
|---|---:|---:|---:|---:|---:|
| No KD | 14.6796 | 33.3831 | 14.9821 | 34.1440 | 24.2972 |
| Vanilla T2/top8, w10 | 13.8745 | 32.2741 | 13.9760 | 33.0875 | 23.3030 |
| KD-BE T2/top8, w10 | 13.4021 | 31.6401 | 13.6850 | 32.3348 | 22.7655 |
| Symmetric n1, w3 | 13.8855 | 32.2191 | 14.4324 | 33.1658 | 23.4257 |
| Guided adapted, w1 | 13.4793 | 31.9993 | 13.7078 | 32.5698 | 22.9391 |
| S-CTC + CTC fine-tune | 12.6392 | 30.7353 | 12.8652 | 31.4025 | 21.9106 |
| CR-CTC, 100 epochs/custom mask | 11.7128 | **28.9295** | 12.2642 | 29.7499 | 20.6641 |
| CR-CTC, current incomplete fair50 | 13.2918 | 30.3918 | 13.5005 | 31.1847 | 22.0922 |
| **Mass25 + NTDK8** | **11.6540** | 29.1454 | **12.0454** | **29.7270** | **20.6430** |

Mass25+NTDK8 beats the 100-epoch CR run on three of four splits and by 0.0211 absolute WER on the four-way average. This is effectively a tie in accuracy, while the CR run uses approximately twice the forward compute. On dev-other CR is better by 0.2159 absolute WER.

Beam gain over greedy averages about 0.317 percentage points for Mass25+NTDK8 and 0.490 for the 100-epoch CR run. Beam evaluation supports the proposed method's final accuracy, but it does not show a uniquely larger beam benefit from NTDK.

