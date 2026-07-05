#!/usr/bin/env python3
"""
Visualize CTC posterior spike timing: Teacher vs Student (No-KD).
At each frame, only the argmax token's probability is drawn (colored by token identity).
Blank frames shown as faint gray dashed line.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav)


def get_probs(model, wav, device):
    wav_t = wav.unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], device=device)
    with torch.no_grad():
        lp, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    T = int(enc_len[0])
    return lp[0, :T].float().exp().cpu().numpy()   # (T, K)


def plot_panel(ax, probs, blank_id, tokenizer, title, token_color):
    """
    Full probability curve per token — one clean colored line each.
    Blank shown as faint gray dashed line.
    """
    T = probs.shape[0]
    frames = np.arange(T)

    # Blank: faint dashed background
    ax.plot(frames, probs[:, blank_id], color="#cccccc", linestyle="--",
            linewidth=1.0, alpha=0.6, label="<blank>", zorder=1)

    # One solid line per non-blank token
    for tok_id, color in token_color.items():
        label = tokenizer.ids_to_tokens([int(tok_id)])[0].replace("▁", "·")
        ax.plot(frames, probs[:, tok_id], color=color, linewidth=1.5,
                label=label, zorder=2)

    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("P(token | frame)", fontsize=8)
    ax.set_xlabel("Frame (10ms steps)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--config", default="configs/student_base.yaml")
    parser.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    parser.add_argument("--out", default="analysis/spike_timing.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading teacher...")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher)
    teacher = teacher.to(device).eval()
    blank_id = teacher.decoder.num_classes_with_blank - 1
    tokenizer = teacher.tokenizer

    print("Loading student...")
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if "test_ds" in model_cfg:
            del model_cfg.test_ds
    student = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    student.load_state_dict(state.get("state_dict", state), strict=False)
    student = student.to(device).eval()

    wav = load_audio(args.audio)

    print("Running inference...")
    t_probs = get_probs(teacher, wav, device)
    s_probs = get_probs(student, wav, device)
    print(f"Teacher frames: {len(t_probs)}, Student frames: {len(s_probs)}")

    # Union of prominent non-blank tokens from both models (by peak probability)
    def top_tokens(probs, blank_id, thresh=0.12, max_n=12):
        nb = probs.copy(); nb[:, blank_id] = 0.0
        peak = nb.max(axis=0)
        cands = np.where(peak > thresh)[0]
        if len(cands) > max_n:
            cands = cands[np.argsort(peak[cands])[::-1]][:max_n]
        return set(int(c) for c in cands)

    all_tokens = sorted(top_tokens(t_probs, blank_id) | top_tokens(s_probs, blank_id))
    print(f"Tokens to plot: {len(all_tokens)}")
    for tok in all_tokens:
        print(f"  {tok}: {tokenizer.ids_to_tokens([tok])[0]}")

    cmap = plt.cm.get_cmap("tab20", max(len(all_tokens), 1))
    token_color = {tok: cmap(i) for i, tok in enumerate(all_tokens)}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    plot_panel(axes[0], t_probs, blank_id, tokenizer,
               f'Teacher (stt_en_conformer_ctc_small)\n"{args.text}"',
               token_color)

    plot_panel(axes[1], s_probs, blank_id, tokenizer,
               "Student — No-KD",
               token_color)

    # Shared legend: collect unique labeled handles from both panels
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    seen, handles, labels = set(), [], []
    for h, l in list(zip(h0, l0)) + list(zip(h1, l1)):
        if l not in seen and not l.startswith("_"):
            seen.add(l)
            handles.append(h)
            labels.append(l)

    fig.legend(handles, labels, loc="center left", fontsize=8,
               bbox_to_anchor=(0.88, 0.5), framealpha=0.9, ncol=1,
               handlelength=1.5, labelspacing=0.6)

    fig.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.09, hspace=0.45)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
