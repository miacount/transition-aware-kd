#!/usr/bin/env python3
"""Figure 1 (ATD) — the blank-suppressed posterior is acoustically grounded.

Two panels, one utterance, shared frame axis:

  (a) blank-suppressed teacher posterior  p_delta6(y_u | t), one row per BPE token
  (b) MFA forced alignment for the same utterance (word intervals, phone splits)

Same y ordering in both panels, so each token's evidence can be read directly
against the interval where that word is actually spoken.

    python3 scripts/fig1_atd_vs_mfa.py [--utt 1188-133604-0014]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))

from analyze_mfa_occupancy import (  # noqa: E402
    blank_penalty_logprobs,
    group_bpe_words,
    read_manifest,
    run_teacher,
)

DELTA = 6.0
C_WORD, C_MUTED, C_GRID = "#eb6834", "#52514e", "#d8d7d2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--alignments", default="data/mfa/test_clean_alignments.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--utt", default="1188-133604-0014")
    ap.add_argument("--out", default="figures/fig1_atd_vs_mfa.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    with open(args.alignments) as f:
        mfa = json.load(f)
    row = next(r for r in read_manifest(args.manifest, 0)
               if Path(r["audio_filepath"]).stem == args.utt)
    align = mfa[args.utt]

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
    toks = tokenizer.ids_to_tokens(ids)
    groups = group_bpe_words(toks)
    words = align["words"]
    assert len(groups) == len(words), (
        f"BPE word grouping ({len(groups)}) != MFA words ({len(words)}); pick another --utt")
    N = len(ids)

    wav, file_sr = sf.read(row["audio_filepath"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert file_sr == sr
    lp = run_teacher(teacher, wav, args.device)
    T = lp.shape[0]
    fd = (len(wav) / sr) / T                     # seconds per frame

    # the ATD evidence: teacher posterior after blank suppression, per transcript token
    post = np.exp(blank_penalty_logprobs(lp, blank, DELTA))[:, ids]      # (T, N)

    tok_word = np.zeros(N, dtype=np.int64)
    for wi, (a, b) in enumerate(groups):
        tok_word[a:b] = wi
    w_start_f = np.array([w["start"] for w in words]) / fd               # -> frames
    w_end_f = np.array([w["end"] for w in words]) / fd

    # share of each token's blank-suppressed mass that sits inside its own word
    frames = np.arange(T) + 0.5
    inside = np.array([
        post[(frames >= w_start_f[tok_word[u]] - 1) &
             (frames < w_end_f[tok_word[u]] + 1), u].sum() / max(post[:, u].sum(), 1e-9)
        for u in range(N)])

    labels = [t.replace("▁", "·") for t in toks]
    f0, f1 = max(0, w_start_f[0] - 6), min(T, w_end_f[-1] + 6)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True,
                             gridspec_kw={"hspace": 0.14})
    fig.patch.set_facecolor("white")

    # ---------------- (a) blank-suppressed posterior -------------------------
    ax = axes[0]
    pcm = ax.pcolormesh(np.arange(T + 1), np.arange(N + 1) - 0.5, post.T, cmap="Blues",
                        norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=1),
                        shading="flat", rasterized=True)
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_ylabel("BPE token", fontsize=11)
    ax.set_title(f"(a)  Blank-suppressed teacher posterior  p̃(y_u | t),  δ = {DELTA:g}",
                 fontsize=11, loc="left", pad=6)
    for u in range(N):
        ax.axhline(u + 0.5, color=C_GRID, lw=0.4)
    cb = fig.colorbar(pcm, ax=ax, pad=0.008, fraction=0.022)
    cb.set_label("p̃(y_u | t)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---------------- (b) MFA forced alignment -------------------------------
    ax = axes[1]
    for u in range(N):
        wi = tok_word[u]
        ax.add_patch(plt.Rectangle((w_start_f[wi], u - 0.34), w_end_f[wi] - w_start_f[wi], 0.68,
                                   facecolor=C_WORD, alpha=0.32, edgecolor=C_WORD, lw=1.1))
    for ph in align["phonemes"]:                 # phone splits, no labels (kept uncluttered)
        s_f, e_f = ph["start"] / fd, ph["end"] / fd
        rows = np.where((w_start_f[tok_word] <= s_f + 1e-6) &
                        (w_end_f[tok_word] >= e_f - 1e-6))[0]
        for u in rows:
            ax.plot([e_f, e_f], [u - 0.34, u + 0.34], color=C_WORD, lw=0.7, alpha=0.8)
    for wi, w in enumerate(words):               # word names on their own ribbon (staggered)
        ax.text(0.5 * (w_start_f[wi] + w_end_f[wi]), -1.5 + 0.55 * (wi % 2), w["word"],
                ha="center", va="center", fontsize=9, color=C_WORD)
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(N - 0.5, -2.0)
    ax.set_ylabel("BPE token", fontsize=11)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_title("(b)  MFA forced alignment  (shaded = word interval, ticks = phone boundaries)",
                 fontsize=11, loc="left", pad=6)
    for u in range(N):
        ax.axhline(u + 0.5, color=C_GRID, lw=0.4)
    fig.colorbar(pcm, ax=ax, pad=0.008, fraction=0.022).ax.set_visible(False)

    for a in axes:
        a.set_xlim(f0, f1)
        a.tick_params(labelsize=9)
        for s in a.spines.values():
            s.set_color(C_GRID)

    # No in-figure title/footnote: the caption carries them in the paper.
    fig.subplots_adjust(left=0.075, right=0.965, top=0.955, bottom=0.085)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor="white")
    print(f"frames={T}  in-word blank-suppressed mass: {100*inside.mean():.1f}%")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
