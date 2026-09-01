# Main-table three-seed protocol (M=32)

The authoritative launcher is `launch_main_table_3seed_m32_tmux.sh`. It
supersedes the earlier LibriSpeech-only `launch_lbs_main_3seed_m32_tmux.sh`.

## Rows

- Teacher: fixed reference; no seed averaging
- No-KD
- Vanilla frame KD
- Symmetric Selection
- Guided CTC
- S-CTC + CTC fine-tuning
- FPKD
- CARL
- CR-CTC
- AT-DKD (ours), top-32 + tail

All student rows use matched seeds 1, 2, and 3 on both LibriSpeech and CHIME-3.
Seed-1 checkpoints are reused; only seeds 2 and 3 are newly trained.

## Staged methods and checkpoint rules

- S-CTC: 80 epochs of S-CTC followed by 20 epochs of CTC fine-tuning.
- FPKD: DFKD 10 + FRKD 10 + PKD 80 epochs.
- CARL: feature stage 10 + full stage 50 epochs. The dev-best checkpoint within
  the 50-epoch full-stage budget is evaluated. Later local 90-epoch CARL
  extensions and last-10 checkpoint averages are intentionally excluded.
- CR-CTC: the frozen stable 50-epoch, two-view, time-mask-factor-1.5 recipe.
- AT-DKD: `L_CTC + 25 L_Mass3 + 8 L_NTDK`, top-32 + tail, 100 epochs.

Every final checkpoint is selected by dev WER and evaluated with no-LM
beam-16 on the four LibriSpeech and four CHIME-3 splits in the paper table.

## Run and resume

```bash
bash experiments/launch_main_table_3seed_m32_tmux.sh
tail -f analysis/main_table_3seed_m32_tmux.log
```

The suite is restart-safe: completed stages, checkpoints, and matching
evaluation logs are reused. Final outputs are:

- `analysis/main_table_3seed_m32_beam16.md`
- `analysis/main_table_3seed_m32_beam16.csv`
