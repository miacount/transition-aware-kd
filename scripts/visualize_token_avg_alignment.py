#!/usr/bin/env python3
"""
Visualize Viterbi forced alignment used for token-avg KD targets.

For each utterance, generates a plot with:
  - Teacher greedy argmax path (colored bar)
  - Viterbi forced alignment (colored bar, blank=grey)
  - Teacher confidence (max prob per frame)
  - Segment annotations (token label, frame count, avg confidence)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BLANK = 0
FLOOR = -1e9


def viterbi_forced_align(log_probs_np, tokens):
    ext = []
    for t in tokens:
        ext.append(BLANK)
        ext.append(t)
    ext.append(BLANK)
    S, T = len(ext), log_probs_np.shape[0]
    alpha = np.full((T, S), FLOOR)
    bp = np.zeros((T, S), dtype=np.int32)
    alpha[0, 0] = log_probs_np[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs_np[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, src = alpha[t - 1, s], s
            if s > 0 and alpha[t - 1, s - 1] > best:
                best, src = alpha[t - 1, s - 1], s - 1
            if s > 1 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                if alpha[t - 1, s - 2] > best:
                    best, src = alpha[t - 1, s - 2], s - 2
            alpha[t, s] = best + log_probs_np[t, ext[s]]
            bp[t, s] = src
    s = S - 1 if alpha[T - 1, S - 1] > alpha[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = s
    for t in range(T - 2, -1, -1):
        s = bp[t + 1, s]
        path[t] = s
    return np.array([ext[s] for s in path], dtype=np.int32)


def get_segments(frame_tokens):
    segs = []
    i = 0
    while i < len(frame_tokens):
        tok = int(frame_tokens[i])
        j = i
        while j < len(frame_tokens) and int(frame_tokens[j]) == tok:
            j += 1
        segs.append({"tok": tok, "start": i, "end": j, "length": j - i})
        i = j
    return segs


def tok_color(tok, blank_id, cmap):
    if tok == blank_id:
        return np.array([0.85, 0.85, 0.85])
    return np.array(cmap(tok % 20)[:3])


def make_figure(row, teacher, tokenizer, blank_id, device, out_path,
                confidence_threshold=0.5, min_seg_len=1):
    wav, sr = sf.read(row["audio_filepath"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        log_probs, enc_len, greedy = teacher.forward(
            input_signal=wav_t, input_signal_length=wav_len
        )

    T = int(enc_len[0].item())
    lp = log_probs[0, :T].float()
    prob = lp.exp().cpu().numpy()           # (T, V)
    lp_np = lp.cpu().numpy()
    greedy_path = greedy[0, :T].cpu().numpy()

    # Viterbi alignment onto teacher greedy tokens
    from build_kd_targets import collapse_repeats
    greedy_seq = collapse_repeats(greedy_path.tolist())
    tokens = [t for t in greedy_seq if t != BLANK]
    viterbi_path = viterbi_forced_align(lp_np, tokens) if tokens else greedy_path.copy()

    segs = get_segments(viterbi_path)
    max_conf = prob.max(axis=-1)            # (T,) teacher confidence per frame
    tok_conf = prob[np.arange(T), np.clip(viterbi_path, 0, prob.shape[1] - 1)]

    cmap = plt.get_cmap("tab20")
    blank_color = np.array([0.85, 0.85, 0.85])

    def path_to_rgb(path):
        rgb = np.zeros((len(path), 3))
        for i, tok in enumerate(path):
            rgb[i] = blank_color if tok == blank_id else np.array(cmap(int(tok) % 20)[:3])
        return rgb

    greedy_rgb  = path_to_rgb(greedy_path)
    viterbi_rgb = path_to_rgb(viterbi_path)

    # mark filtered segments in viterbi bar (hatched with red tint)
    viterbi_filtered_rgb = viterbi_rgb.copy()
    for seg in segs:
        if seg["tok"] == blank_id:
            continue
        tok = seg["tok"]
        length = seg["length"]
        avg_p = prob[seg["start"]:seg["end"]].mean(axis=0)
        conf = avg_p[tok]
        if length < min_seg_len or conf < confidence_threshold:
            # filtered: red tint
            viterbi_filtered_rgb[seg["start"]:seg["end"]] = [0.9, 0.4, 0.4]

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 6),
        gridspec_kw={"height_ratios": [1, 1, 1.5, 1.5], "hspace": 0.55}
    )

    def imrow(ax, rgb, title):
        ax.imshow(rgb[np.newaxis], aspect="auto", interpolation="nearest",
                  extent=[0, T, 0, 1])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=8, pad=2)

    imrow(axes[0], greedy_rgb,
          f"Teacher greedy argmax  (blank=grey, T={T})")
    imrow(axes[1], viterbi_rgb,
          "Viterbi forced alignment  (non-blank segments colored by token)")
    imrow(axes[2], viterbi_filtered_rgb,
          f"Viterbi alignment after filter  "
          f"(red=filtered: conf<{confidence_threshold} or len<{min_seg_len})")

    # segment annotations on viterbi row
    nonblank_segs = [s for s in segs if s["tok"] != blank_id]
    for seg in nonblank_segs:
        mid = (seg["start"] + seg["end"]) / 2
        avg_p = prob[seg["start"]:seg["end"]].mean(axis=0)
        conf = avg_p[seg["tok"]]
        label = tokenizer.ids_to_text([seg["tok"]])
        filtered = seg["length"] < min_seg_len or conf < confidence_threshold
        color = "red" if filtered else "black"
        axes[1].axvline(seg["start"], color="white", lw=0.4, alpha=0.5)
        axes[1].text(mid, 1.08, f"{label}\n{seg['length']}f\n{conf:.2f}",
                     ha="center", va="bottom", fontsize=5.5, color=color,
                     transform=axes[1].get_xaxis_transform())

    # confidence plot
    ax_conf = axes[3]
    ax_conf.fill_between(np.arange(T), max_conf, alpha=0.35, color="#4477AA",
                         label="max prob (any token)")
    ax_conf.fill_between(np.arange(T), tok_conf, alpha=0.6, color="#EE6633",
                         label="prob of aligned token")
    ax_conf.axhline(confidence_threshold, color="red", lw=0.8, ls="--",
                    label=f"conf threshold ({confidence_threshold})")
    for seg in nonblank_segs:
        ax_conf.axvspan(seg["start"], seg["end"], alpha=0.08, color="green")
    ax_conf.set_xlim(0, T)
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_xlabel("encoder frame index", fontsize=8)
    ax_conf.set_ylabel("probability", fontsize=8)
    ax_conf.set_title("Teacher confidence per frame  (green shading = non-blank segment)",
                      fontsize=8, pad=2)
    ax_conf.legend(fontsize=7, loc="upper right")

    # summary stats
    nb_segs = nonblank_segs
    kept = [s for s in nb_segs
            if s["length"] >= min_seg_len
            and prob[s["start"]:s["end"]].mean(axis=0)[s["tok"]] >= confidence_threshold]
    blank_frames = sum(1 for t in viterbi_path if t == blank_id)
    nb_1f = sum(1 for s in nb_segs if s["length"] == 1)

    text_ref = row.get("text", "")[:80]
    title = (
        f"\"{text_ref}\"\n"
        f"T={T}  |  non-blank segs: {len(nb_segs)}  |  1-frame segs: {nb_1f} ({100*nb_1f/max(len(nb_segs),1):.0f}%)  "
        f"|  blank frames: {blank_frames} ({100*blank_frames/T:.0f}%)  |  kept: {len(kept)}/{len(nb_segs)}"
    )
    fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train_clean_100.small_teacher.transition.json")
    ap.add_argument("--teacher",  default="stt_en_conformer_ctc_small")
    ap.add_argument("--n",        type=int, default=5, help="number of utterances to plot")
    ap.add_argument("--out_dir",  default="analysis/alignment_viz")
    ap.add_argument("--confidence_threshold", type=float, default=0.5)
    ap.add_argument("--min_seg_len",          type=int,   default=2)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--offset",   type=int, default=0, help="start from this utterance index")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    # need collapse_repeats from build_kd_targets
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.teacher}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher)
    teacher = teacher.to(args.device).eval()
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    tokenizer = teacher.tokenizer

    rows = []
    with open(args.manifest) as f:
        for i, line in enumerate(f):
            if line.strip():
                rows.append(json.loads(line))

    rows = rows[args.offset: args.offset + args.n]
    print(f"plotting {len(rows)} utterances (offset={args.offset})")

    for idx, row in enumerate(rows):
        utt_id = Path(row["audio_filepath"]).stem
        out_path = out_dir / f"tokavg_{args.offset+idx:04d}_{utt_id}.png"
        make_figure(row, teacher, tokenizer, blank_id, torch.device(args.device),
                    out_path,
                    confidence_threshold=args.confidence_threshold,
                    min_seg_len=args.min_seg_len)


if __name__ == "__main__":
    main()
