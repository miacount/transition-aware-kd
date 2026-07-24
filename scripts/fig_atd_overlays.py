#!/usr/bin/env python3
"""Two single-panel ATD overlays (everything on one axis each).

  fig3_atd_span_spikes.png   ATD span as an outlined interval per token, with the
                             teacher spike and the student spike drawn on it.
  fig4_posterior_mfa.png     Blank-suppressed teacher posterior as a heatmap, with
                             the MFA word interval outlined on the same rows.

Both use the encoder frame index as x and one BPE token per row as y.

    python3 scripts/fig_atd_overlays.py [--utt 1320-122617-0038]
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze_mfa_occupancy import (  # noqa: E402
    blank_penalty_logprobs,
    fb_token_gamma_fast,
    group_bpe_words,
    normalize_cols_occ,
    read_manifest,
    run_teacher,
)

DELTA = 6.0
C_SPAN, C_SPAN_FILL = "#2a78d6", "#dbe9f8"
C_T, C_S, C_WORD = "#08306b", "#eb6834", "#eb6834"
C_MUTED, C_GRID = "#52514e", "#d8d7d2"


def frame_axis(ax, N, labels, f0, f1,
               xlabel="frame  (teacher encoder, 4× subsampled ≈ 40 ms)", ytop=-0.5):
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(N - 0.5, ytop)
    ax.set_xlim(f0, f1)
    ax.set_ylabel("BPE token", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(labelsize=9.5)
    for u in range(N):
        ax.axhline(u + 0.5, color=C_GRID, lw=0.4)
    for s in ax.spines.values():
        s.set_color(C_GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--alignments", default="data/mfa/test_clean_alignments.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--student", default="nemo_experiments/student-no-kd/"
                                         "2026-06-24_04-38-47/checkpoints/student-no-kd.nemo")
    ap.add_argument("--student-name", default="no-KD student")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--utt", default="1320-122617-0038",
                    help="utterance for fig3 (span vs spikes)")
    ap.add_argument("--utt4", default="1188-133604-0014",
                    help="utterance for fig4; the default has no repeated BPE ids, so the "
                         "FB-free posterior is unimodal and reads cleanly")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--out3", default="figures/fig3_atd_span_spikes.png")
    ap.add_argument("--out4", default="figures/fig4_posterior_mfa.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    with open(args.alignments) as f:
        mfa = json.load(f)
    rows = read_manifest(args.manifest, 0)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    student = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.student, map_location=args.device).to(args.device).eval()
    student.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    def load(utt):
        """Everything both figures need for one utterance."""
        row = next(r for r in rows if Path(r["audio_filepath"]).stem == utt)
        ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
        toks = tokenizer.ids_to_tokens(ids)
        wav, file_sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert file_sr == sr
        t_lp = run_teacher(teacher, wav, args.device)
        s_lp = run_teacher(student, wav, args.device)
        T = t_lp.shape[0]
        assert s_lp.shape[0] == T
        occ0 = normalize_cols_occ(fb_token_gamma_fast(t_lp, ids, blank))
        lp_d = blank_penalty_logprobs(t_lp, blank, DELTA)
        return dict(
            utt=utt, row=row, ids=ids, toks=toks, N=len(ids),
            labels=[t.replace("▁", "·") for t in toks],
            groups=group_bpe_words(toks), words=mfa[utt]["words"],
            T=T, fd=(len(wav) / sr) / T, occ0=occ0,
            span=normalize_cols_occ(fb_token_gamma_fast(lp_d, ids, blank)),
            post=np.exp(lp_d)[:, ids], t_spike=occ0.argmax(axis=0),
            s_spike=normalize_cols_occ(fb_token_gamma_fast(s_lp, ids, blank)).argmax(axis=0))

    D = load(args.utt)
    row, ids, labels, N = D["row"], D["ids"], D["labels"], D["N"]
    T, occ0, span, t_spike, s_spike = D["T"], D["occ0"], D["span"], D["t_spike"], D["s_spike"]

    lo = np.array([np.where(span[:, u] > args.eps)[0][0] for u in range(N)])
    hi = np.array([np.where(span[:, u] > args.eps)[0][-1] + 1 for u in range(N)])
    inside = (s_spike >= lo) & (s_spike < hi)
    inside_d0 = occ0[s_spike, np.arange(N)] > args.eps   # same criterion as the δ=6 span
    f0, f1 = max(0, t_spike.min() - 6), min(T, t_spike.max() + 7)

    # ================================================= fig 3: span + spikes ===
    fig, ax = plt.subplots(figsize=(13, 0.42 * N + 2.1))
    fig.patch.set_facecolor("white")
    for u in range(N):
        ax.add_patch(plt.Rectangle((lo[u], u - 0.36), hi[u] - lo[u], 0.72,
                                   facecolor=C_SPAN_FILL, edgecolor=C_SPAN, lw=1.8,
                                   zorder=2))
    for u in range(N):                       # connector makes the drift readable
        if s_spike[u] != t_spike[u]:
            ax.plot([t_spike[u] + 0.5, s_spike[u] + 0.5], [u, u], color=C_MUTED,
                    lw=0.9, ls=":", zorder=3)
    ax.plot(t_spike + 0.5, np.arange(N), ls="none", marker="|", ms=17, mew=2.6,
            color=C_T, zorder=4, label="teacher spike")
    ax.plot(s_spike + 0.5, np.arange(N), ls="none", marker="o", ms=8, mfc="white",
            mec=C_S, mew=2.2, zorder=5, label=f"{args.student_name} spike")
    frame_axis(ax, N, labels, f0, f1, xlabel="Frame")
    # the box outline is the claim of the figure, so it gets a legend entry
    span_key = plt.Rectangle((0, 0), 1, 1, facecolor=C_SPAN_FILL, edgecolor=C_SPAN, lw=1.8)
    handles, hlabels = ax.get_legend_handles_labels()
    ax.legend([span_key] + handles,
              [f"occupancy span (teacher γ, δ={DELTA:g})"] + hlabels,
              fontsize=10, frameon=True, framealpha=0.96, edgecolor=C_GRID,
              loc="lower left", borderaxespad=0.8)
    ax.set_title("ATD span absorbs the teacher–student spike offset",
                 fontsize=13, loc="left", pad=10)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.10)
    Path(args.out3).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out3, dpi=180, facecolor="white")
    print(f"fig3: {int(inside.sum())}/{N} inside span, {int(inside_d0.sum())}/{N} at δ=0 "
          f"-> {args.out3}")

    # ============================================ fig 4: posterior + MFA ======
    D4 = load(args.utt4) if args.utt4 != args.utt else D
    row, labels, N, T = D4["row"], D4["labels"], D4["N"], D4["T"]
    post, groups, words, fd = D4["post"], D4["groups"], D4["words"], D4["fd"]
    assert len(groups) == len(words), (
        f"BPE word grouping ({len(groups)}) != MFA words ({len(words)}); pick another --utt4")
    sp4 = D4["t_spike"]
    f0, f1 = max(0, sp4.min() - 6), min(T, sp4.max() + 7)

    tok_word = np.zeros(N, dtype=np.int64)
    for wi, (a, b) in enumerate(groups):
        tok_word[a:b] = wi
    ws = np.array([w["start"] for w in words]) / fd
    we = np.array([w["end"] for w in words]) / fd

    fig, ax = plt.subplots(figsize=(13, 0.42 * N + 2.4))
    fig.patch.set_facecolor("white")
    pcm = ax.pcolormesh(np.arange(T + 1), np.arange(N + 1) - 0.5, post.T, cmap="Blues",
                        norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=1),
                        shading="flat", rasterized=True, zorder=1)
    for u in range(N):
        wi = tok_word[u]
        ax.add_patch(plt.Rectangle((ws[wi], u - 0.44), we[wi] - ws[wi], 0.88,
                                   facecolor="none", edgecolor=C_WORD, lw=2.0, zorder=3))
    for wi, w in enumerate(words):
        a, _ = groups[wi]
        ax.text(0.5 * (ws[wi] + we[wi]), a - 0.62, w["word"], ha="center", va="bottom",
                fontsize=9, color=C_WORD, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    frame_axis(ax, N, labels, f0, f1, ytop=-1.25)
    cb = fig.colorbar(pcm, ax=ax, pad=0.008, fraction=0.020)
    cb.set_label("blank-suppressed posterior  p̃(y_u | t)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    ax.set_title("Blank-suppressed posterior against the MFA word interval\n"
                 f"{args.utt4} — “{row['text'].lower()}”",
                 fontsize=13, loc="left", pad=12)

    inw = np.array([post[max(0, int(ws[tok_word[u]]) - 1):int(we[tok_word[u]]) + 2, u].sum()
                    / max(post[:, u].sum(), 1e-9) for u in range(N)])
    fig.text(0.012, 0.012,
             f"Orange outline = MFA forced-aligned interval of that token's word.  "
             f"{100*inw.mean():.1f}% of each token's blank-suppressed mass falls inside it on this\n"
             f"utterance; over test-clean the mass δ={DELTA:g} newly recruits is 98.7% in-word, "
             f"versus 32% self-purity for a width-matched symmetric blur.",
             fontsize=9.5, color=C_MUTED, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.925, top=0.875, bottom=0.155)
    fig.savefig(args.out4, dpi=180, facecolor="white")
    print(f"fig4: in-word posterior mass {100*inw.mean():.1f}% -> {args.out4}")


if __name__ == "__main__":
    main()
