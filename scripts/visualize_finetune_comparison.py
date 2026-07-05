#!/usr/bin/env python3
"""Compare the original teacher vs a de-peak fine-tuned teacher, side by side.

For each utterance shows:
  col 0: log-mel
  col 1: original teacher BPE occupancy gamma(t,u)   (peaky)
  col 2: fine-tuned teacher BPE occupancy gamma(t,u) (de-peaked)
  col 3: per-frame P(blank) over time, both models

Reports P(blank) and mean effective token duration for both.

Example:
  python scripts/visualize_finetune_comparison.py \
    --finetuned nemo_experiments/teacher_depeak_d0.5/teacher_depeak.nemo \
    --manifest data/dev_clean.json --n 4 \
    --out analysis/bpe_aligner/finetune_d0.5_compare.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from visualize_ctc_forward_backward import (  # noqa: E402
    ctc_forward_backward, load_audio, plot_mel, read_manifest, token_labels,
)


@torch.no_grad()
def run(model, wav, device):
    wt = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    wl = torch.tensor([wt.shape[1]], dtype=torch.long, device=device)
    lp, enc_len, _ = model.forward(input_signal=wt, input_signal_length=wl)
    return lp[0, :int(enc_len[0])].detach().cpu().float().numpy()


def eff_dur(tg):
    s1 = tg.sum(0); s2 = (tg ** 2).sum(0)
    e = (s1 ** 2) / np.maximum(s2, 1e-12)
    v = s1 > 1e-6
    return float(e[v].mean()) if v.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True, help="path to fine-tuned .nemo")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="analysis/bpe_aligner/finetune_compare.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    orig = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    orig.freeze()
    ft = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.finetuned).to(args.device).eval()
    ft.freeze()
    blank = int(orig.decoder.num_classes_with_blank - 1)
    sr = int(orig.cfg.preprocessor.sample_rate)
    tok = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    rows = read_manifest(args.manifest, n=args.n, offset=args.offset)
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 3.2 * len(rows)), squeeze=False)

    print(f"{'utt':>3} {'model':>10} {'P(blank)':>9} {'eff_dur':>8}")
    for r, row in enumerate(rows):
        wav = load_audio(row["audio_filepath"], sr)
        dur = len(wav) / float(sr)
        ids = [int(i) for i in tok.text_to_ids(row["text"]) if int(i) != blank]
        labels = token_labels(tok, ids)

        lp_o = run(orig, wav, args.device)
        lp_f = run(ft, wav, args.device)
        fb_o = ctc_forward_backward(lp_o, ids, blank)
        fb_f = ctc_forward_backward(lp_f, ids, blank)
        tg_o, tg_f = fb_o["token_gamma"], fb_f["token_gamma"]

        for name, fb, tg in [("original", fb_o, tg_o), ("finetuned", fb_f, tg_f)]:
            print(f"{r:>3} {name:>10} {fb['blank_gamma'].mean():>9.3f} {eff_dur(tg):>8.2f}")

        plot_mel(axes[r][0], wav, sr, dur)
        axes[r][0].set_ylabel(f"utt {r}", fontsize=8)
        if r == 0:
            axes[r][0].set_title("mel", fontsize=9)

        for c, (title, tg) in enumerate([("original γ (peaky)", tg_o),
                                         ("finetuned γ (de-peaked)", tg_f)]):
            ax = axes[r][c + 1]
            ax.imshow(tg.T, origin="lower", aspect="auto",
                      extent=[0, dur, -0.5, len(ids) - 0.5], interpolation="nearest",
                      cmap="viridis", vmin=0.0, vmax=max(0.3, float(np.quantile(tg, 0.999))))
            if r == 0:
                ax.set_title(title, fontsize=9)
            ax.set_xlabel("time (s)", fontsize=7)
            if len(labels) <= 40:
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels, fontsize=4)
            else:
                ax.set_yticks([])

        # per-frame blank probability over time
        ax = axes[r][3]
        t_o = np.linspace(0, dur, lp_o.shape[0])
        t_f = np.linspace(0, dur, lp_f.shape[0])
        ax.plot(t_o, np.exp(lp_o[:, blank]), color="#CC6677", lw=1.0, label="original")
        ax.plot(t_f, np.exp(lp_f[:, blank]), color="#4477AA", lw=1.0, label="finetuned")
        ax.set_ylim(-0.02, 1.02)
        if r == 0:
            ax.set_title("P(blank) per frame", fontsize=9)
            ax.legend(fontsize=6, loc="lower right")
        ax.set_xlabel("time (s)", fontsize=7)
        ax.grid(alpha=0.2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
