#!/usr/bin/env python3
"""Diagnose a teacher-only BPE-domain soft aligner with blank suppression.

Idea (A+C unified): build the BPE-level soft alignment directly from the
teacher's own CTC posterior, de-peaked by subtracting a scalar log-penalty
delta from ONLY the blank log-probability. Equivalently this multiplies every
non-blank token by the same factor, so:

  - non-blank / non-blank ratios are preserved exactly  -> dark knowledge kept
  - the boost is identical for every non-blank token     -> no rare-BPE tail
  - freed blank mass is redistributed proportional to the teacher's existing
    posterior (NOT uniformly), so it flows to the tokens the teacher already
    favors -> meaningful, never random

    logp'_t[blank]     = logp_t[blank] - delta   # delta >= 0 (nats)
    logp'_t[non-blank] = logp_t[non-blank]       # untouched
    gamma              = ctc_forward_backward(logp', bpe_ids, blank)

The CTC forward-backward is transcript-constrained: only this utterance's BPE
tokens + blank appear in valid paths, so the widened occupancy lands ONLY on
the real transcript tokens. delta = 0 recovers the current peaky gamma mode.

For each utterance and each delta we plot the BPE token occupancy gamma(t,u)
and report:
  - blank_occ : mean blank occupancy over frames (want it to drop with delta)
  - eff_dur   : mean effective token duration in frames (participation ratio;
                want it to grow above ~1.0 = de-peaking)
  - dark_KL   : KL(q_u at this delta || q_u at delta=0), mean over tokens
                (want small: semantic target / dark knowledge preserved)
  - transcript_share : fraction of the per-frame non-blank mass that lands on
                THIS transcript's tokens after suppression (want it high =
                freed mass goes to meaningful tokens, not garbage vocab)

Example:
  python scripts/visualize_bpe_soft_aligner.py \
    --manifest data/dev_clean.json --n 4 \
    --deltas 0 2 4 8 \
    --out analysis/bpe_aligner/dev_clean_delta.png
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from visualize_ctc_forward_backward import (  # noqa: E402
    ctc_forward_backward,
    load_audio,
    plot_mel,
    read_manifest,
    token_labels,
)

NEG_INF = -1.0e30


def logsumexp_lastdim(a):
    m = np.max(a, axis=-1, keepdims=True)
    m = np.where(m <= NEG_INF / 2, 0.0, m)
    return m.squeeze(-1) + np.log(np.exp(a - m).sum(axis=-1))


def blank_penalty_logprobs(log_probs, blank_id, delta):
    """Subtract delta (nats) from blank log-prob only; renormalize per frame.
    Non-blank log-probs are untouched, so their ratios are exactly preserved."""
    if delta == 0.0:
        return log_probs
    lp = log_probs.copy()
    lp[:, blank_id] = log_probs[:, blank_id] - delta
    lp = lp - logsumexp_lastdim(lp)[:, None]
    return lp


@torch.no_grad()
def run_teacher(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


def semantic_target(log_probs, token_gamma, blank_id):
    """q_u = normalize_over_vocab( sum_t gamma(t,u) * p_t ), blank zeroed. (N, V)."""
    probs = np.exp(log_probs)
    q = token_gamma.T @ probs
    q[:, blank_id] = 0.0
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    return q


def kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * np.log(p / q)).sum(axis=1).mean())


def effective_duration(token_gamma):
    s1 = token_gamma.sum(axis=0)
    s2 = (token_gamma ** 2).sum(axis=0)
    eff = (s1 ** 2) / np.maximum(s2, 1e-12)
    valid = s1 > 1e-6
    return float(eff[valid].mean()) if valid.any() else 0.0


def transcript_mass_share(log_probs, bpe_ids, blank_id):
    """Of the non-blank probability mass per frame, what fraction sits on THIS
    transcript's unique tokens? High = freed mass goes to meaningful tokens."""
    probs = np.exp(log_probs)
    nonblank = probs.copy()
    nonblank[:, blank_id] = 0.0
    total_nb = nonblank.sum(axis=1)
    uniq = np.array(sorted(set(int(i) for i in bpe_ids)), dtype=np.int64)
    on_transcript = nonblank[:, uniq].sum(axis=1)
    valid = total_nb > 1e-8
    return float((on_transcript[valid] / total_nb[valid]).mean()) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0, 2, 4, 8],
                    help="blank log-penalty in nats; 0 = unmodified peaky gamma")
    ap.add_argument("--out", default="analysis/bpe_aligner/delta_sweep.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest, n=args.n, offset=args.offset)
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    deltas = args.deltas
    n_cols = 1 + len(deltas)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(4 * n_cols, 3.0 * len(rows)),
                             squeeze=False)

    print(f"{'utt':>3} {'delta':>5} {'blank_occ':>9} {'eff_dur':>8} "
          f"{'dark_KL':>8} {'transcript_share':>16}")
    for r, row in enumerate(rows):
        wav = load_audio(row["audio_filepath"], sample_rate)
        duration = len(wav) / float(sample_rate)
        log_probs = run_teacher(teacher, wav, args.device)
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != blank_id]
        labels = token_labels(tokenizer, bpe_ids)

        plot_mel(axes[r][0], wav, sample_rate, duration)
        axes[r][0].set_ylabel(f"utt {r}", fontsize=8)
        if r == 0:
            axes[r][0].set_title("mel", fontsize=9)

        fb_ref = ctc_forward_backward(log_probs, bpe_ids, blank_id)
        q_ref = semantic_target(log_probs, fb_ref["token_gamma"], blank_id)

        for c, delta in enumerate(deltas):
            lp = blank_penalty_logprobs(log_probs, blank_id, delta)
            fb = ctc_forward_backward(lp, bpe_ids, blank_id)
            tg = fb["token_gamma"]
            blank_occ = float(fb["blank_gamma"].mean())
            eff = effective_duration(tg)
            q = semantic_target(lp, tg, blank_id)
            dark_kl = kl(q, q_ref)
            tshare = transcript_mass_share(lp, bpe_ids, blank_id)
            print(f"{r:>3} {delta:>5.1f} {blank_occ:>9.4f} {eff:>8.2f} "
                  f"{dark_kl:>8.4f} {tshare:>16.4f}")

            ax = axes[r][c + 1]
            ax.imshow(tg.T, origin="lower", aspect="auto",
                      extent=[0, duration, -0.5, len(bpe_ids) - 0.5],
                      interpolation="nearest", cmap="viridis",
                      vmin=0.0, vmax=max(0.3, float(np.quantile(tg, 0.999))))
            if r == 0:
                ax.set_title(f"γ  δ={delta}", fontsize=9)
            ax.set_xlabel("time (s)", fontsize=7)
            if len(labels) <= 40:
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels, fontsize=4)
            else:
                ax.set_yticks([])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
