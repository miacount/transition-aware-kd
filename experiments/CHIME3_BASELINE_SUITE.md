# CHiME-3 sequential baseline suite

All experiments use only the original-speed enhanced mono manifests. The
teacher, dense targets, no-KD result, and proposed-method result are reused.

## Shared controls

- Student: Conformer 144 x 8, subsampling 4, seed 1.
- Optimizer: Noam coefficient 0.5, 750-step warm-up unless a staged method
  explicitly uses a shorter dataset-scaled restart.
- Selection: minimum combined CHiME dev WER; eval splits are report-only.
- Decoding: greedy during training and Beam-16 after the complete queue.
- Safety: standalone config checks, resumable target generation, five-epoch
  convergence gates, per-stage last-checkpoint continuation, and isolated
  evaluation logs.

## Methods

| Method | Schedule | Fixed method settings |
|---|---:|---|
| Vanilla frame KD | 100 | T=1, utterance-sum, lambda=.25 |
| Blank Elimination KD | 100 | Vanilla + eliminate blank frames |
| Symmetric KD | 100 | Vanilla + symmetric n=4 |
| Guided CTC | 100 | exact guided loss weight 1 |
| S-CTC + CTC FT | 80+20 | pure S-CTC; FT lr=.05/warm-up=100 |
| FPKD | 10+10+80 | DFKD, FRKD, PKD; dataset-scaled warm-ups |
| CARL | 10+50 | feature/full CTC stages; average final 10 epochs |
| CR-CTC | 50 dual-view | alpha=.2, adapted mask factor 1.5 |

The baseline-specific values are transferred from the frozen LibriSpeech
protocol rather than selected on CHiME eval. CR-CTC deliberately uses the
locally validated NeMo adaptation: the literal 2.5x count-and-width port is
excluded because it previously collapsed this architecture.
