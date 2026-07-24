#!/usr/bin/env python3
"""Figure 2 (ATD) — spike misalignment is absorbed by the ATD span.

Three panels, one utterance, shared time axis:

  (a) teacher raw posterior          -- where the teacher fires each token
  (b) student raw posterior (no-KD)  -- the student fires the same tokens elsewhere
  (c) ATD span (delta=6 occupancy)   -- both models' spikes drawn on the target
                                        support: the student's spike is inside it

Panel (b) also carries the teacher's spike positions as faint reference lines, so
the per-token offset is readable directly.

    python3 scripts/fig2_atd_misalignment.py [--utt 1089-134686-0001]
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
    normalize_cols_occ,
    read_manifest,
    run_teacher,
)

DELTA = 6.0
C_T, C_S, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#52514e", "#d8d7d2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--student", default="nemo_experiments/student-no-kd/"
                                         "2026-06-24_04-38-47/checkpoints/student-no-kd.nemo")
    ap.add_argument("--student-name", default="no-KD student")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--utt", default="1320-122617-0038")
    ap.add_argument("--eps", type=float, default=0.01, help="occupancy threshold for 'inside'")
    ap.add_argument("--coverage-json", default="analysis/spike_coverage/summary.json")
    ap.add_argument("--out", default="figures/fig2_atd_misalignment.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    row = next(r for r in read_manifest(args.manifest, 0)
               if Path(r["audio_filepath"]).stem == args.utt)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    student = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.student, map_location=args.device).to(args.device).eval()
    student.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
    labels = [t.replace("▁", "·") for t in tokenizer.ids_to_tokens(ids)]
    N = len(ids)

    wav, file_sr = sf.read(row["audio_filepath"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert file_sr == sr
    t_lp = run_teacher(teacher, wav, args.device)
    s_lp = run_teacher(student, wav, args.device)
    T = t_lp.shape[0]
    assert s_lp.shape[0] == T, "teacher/student frame rates differ"
    frames = np.arange(T) + 0.5          # x axis is the encoder frame index

    occ0 = normalize_cols_occ(fb_token_gamma_fast(t_lp, ids, blank))
    span = normalize_cols_occ(fb_token_gamma_fast(
        blank_penalty_logprobs(t_lp, blank, DELTA), ids, blank))
    t_spike = occ0.argmax(axis=0)
    s_spike = normalize_cols_occ(fb_token_gamma_fast(s_lp, ids, blank)).argmax(axis=0)
    off = s_spike - t_spike

    cov_span = span[s_spike, np.arange(N)]
    cov_d0 = occ0[s_spike, np.arange(N)]
    in_span = int((cov_span > args.eps).sum())
    in_d0 = int((cov_d0 > args.eps).sum())

    f0, f1 = max(0, t_spike.min() - 6), min(T, t_spike.max() + 7)

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.9], "hspace": 0.20})
    fig.patch.set_facecolor("white")

    def post_panel(ax, lp, spike, color, title, ref_spike=None):
        p = np.exp(lp)
        ax.fill_between(frames, 1.0 - p[:, blank], color=color, alpha=0.85, lw=0, zorder=3)
        if ref_spike is not None:
            for u in range(N):
                ax.axvline(frames[ref_spike[u]], color=C_T, lw=1.0, ls=":", alpha=0.75, zorder=2)
        for u in range(N):
            ax.axvline(frames[spike[u]], color=color, lw=0.9, alpha=0.45, zorder=1)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("1 − p(blank)", fontsize=11)
        ax.set_title(title, fontsize=12, loc="left", pad=8)
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)

    post_panel(axes[0], t_lp, t_spike, C_T, "(a)  Teacher raw posterior")
    post_panel(axes[1], s_lp, s_spike, C_S,
               f"(b)  Student raw posterior — {args.student_name}   "
               f"(dotted = teacher spike positions; mean |offset| on this utterance "
               f"= {np.abs(off).mean():.2f} frames)", ref_spike=t_spike)

    # ---------------- (c) ATD span with both spikes --------------------------
    ax = axes[2]
    pcm = ax.pcolormesh(np.arange(T + 1), np.arange(N + 1) - 0.5, span.T, cmap="Blues",
                        norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=1),
                        shading="flat", rasterized=True)
    for u in range(N):                       # outline the extent of each token's span
        on = np.where(span[:, u] > args.eps)[0]
        if len(on):
            ax.add_patch(plt.Rectangle((on[0], u - 0.42), on[-1] + 1 - on[0], 0.84,
                                       facecolor="none", edgecolor="#08306b", lw=1.4))
        ax.axhline(u + 0.5, color=C_GRID, lw=0.4)
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_ylabel("BPE token", fontsize=11)
    ax.set_xlabel("frame  (teacher encoder, 4× subsampled ≈ 40 ms)", fontsize=11)
    ax.set_title("(c)  ATD span — teacher occupancy γ (δ=6), the KD target support "
                 "(outline = span extent)", fontsize=12, loc="left", pad=8)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(pcm, ax=ax, pad=0.008, fraction=0.022)
    cb.set_label("occupancy γ(t, u)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    for a in axes[:2]:
        fig.colorbar(pcm, ax=a, pad=0.008, fraction=0.022).ax.set_visible(False)

    for a in axes:
        a.set_xlim(f0, f1)

    fig.suptitle("Student spikes drift from the teacher's — the ATD span catches "
                 "twice as many as a frame-level target",
                 fontsize=14, x=0.075, ha="left", y=0.985)
    fig.text(0.075, 0.952, f"{args.utt} — “{row['text'].lower()}”",
             fontsize=10.5, color=C_MUTED, ha="left")

    cap2 = cap3 = ""
    if os.path.exists(args.coverage_json):
        S = json.load(open(args.coverage_json))["students"]
        k = next((n for n in S if "no-KD" in n), None)
        if k:
            st = S[k]["by_stratum"]
            cap2 = (f"Over test-clean (8,237 tokens): {100*(1-st['0']['share']):.1f}% of tokens are "
                    f"misaligned, and {100*st['1']['n']/(st['all']['n']-st['0']['n']):.0f}% of those "
                    f"sit at exactly ±1 frame.")
            cap3 = (f"On those ±1 tokens a frame-KD target (δ=0) puts mass on the student's frame "
                    f"{100*st['1']['delta=0']['hit']:.1f}% of the time; the ATD span, "
                    f"{100*st['1']['delta=6']['hit']:.1f}%.")
    fig.text(0.075, 0.052,
             f"This utterance: {in_span}/{N} student spikes fall inside the ATD span; "
             f"a δ=0 (frame-KD) target would cover only {in_d0}/{N}.",
             fontsize=10, color=C_MUTED)
    fig.text(0.075, 0.028, cap2, fontsize=10, color=C_MUTED)
    fig.text(0.075, 0.006, cap3, fontsize=10, color=C_MUTED)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.918, bottom=0.118)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor="white")
    print(f"mean|offset|={np.abs(off).mean():.2f}  inside ATD span {in_span}/{N}  "
          f"inside δ=0 {in_d0}/{N}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
