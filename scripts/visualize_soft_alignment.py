#!/usr/bin/env python3
"""
Compare alignment posteriors: peaky Conformer teacher vs soft TDNN-FFN aligner.

For each sample utterance:
  row 1: waveform + mel spectrogram
  row 2: Conformer posteriorgram (only transcript tokens + blank)
  row 3: TDNN-FFN soft aligner posteriorgram
  Viterbi path overlaid on each posteriorgram.

Usage:
  python scripts/visualize_soft_alignment.py \
    --soft-ckpt nemo_experiments/soft_aligner/best.pt \
    --manifest  data/dev_clean.json \
    --n 4 --out plots/soft_alignment.png
"""
import os
import sys
import json
import argparse

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# reuse model class
from train_soft_aligner import SoftAligner


# ---------------------------------------------------------------------------
# Forced alignment (Viterbi)
# ---------------------------------------------------------------------------

def viterbi_align(log_probs_np: np.ndarray, tokens: list, blank_id: int) -> np.ndarray:
    """Returns per-frame label sequence (hard Viterbi alignment)."""
    FLOOR = -1e9
    ext = []
    for t in tokens:
        ext.append(blank_id)
        ext.append(t)
    ext.append(blank_id)
    S = len(ext); T = log_probs_np.shape[0]
    alpha = np.full((T, S), FLOOR)
    bp    = np.zeros((T, S), dtype=np.int32)
    alpha[0, 0] = log_probs_np[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs_np[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, src = alpha[t-1, s], s
            if s > 0 and alpha[t-1, s-1] > best:
                best, src = alpha[t-1, s-1], s-1
            if (s > 1 and ext[s] != blank_id and ext[s] != ext[s-2]
                    and alpha[t-1, s-2] > best):
                best, src = alpha[t-1, s-2], s-2
            alpha[t, s] = best + log_probs_np[t, ext[s]]
            bp[t, s] = src
    s = S-1 if alpha[T-1, S-1] > alpha[T-1, S-2] else S-2
    path = np.zeros(T, dtype=np.int32)
    path[T-1] = s
    for t in range(T-2, -1, -1):
        s = bp[t+1, s]
        path[t] = s
    return np.array([ext[s] for s in path], dtype=np.int32)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_conformer(teacher_name: str, device: str):
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(teacher_name)
    return model.to(device).eval()


def load_soft_aligner(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = SoftAligner(input_dim=80, hidden=a.get("hidden", 512),
                        vocab=1025)
    model.load_state_dict(ckpt["state_dict"])
    log_priors = ckpt["log_priors"]
    return model.to(device).eval(), log_priors


def get_mel(wav_np: np.ndarray) -> torch.Tensor:
    """Log-mel matching NeMo preprocessor, pure librosa (no CUDA)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=wav_np, sr=16000, n_fft=512, win_length=400, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, power=2.0,
    )
    mel = np.log(mel + 1e-6)
    mean = mel.mean(axis=1, keepdims=True)
    std  = mel.std(axis=1,  keepdims=True)
    mel  = (mel - mean) / (std + 1e-5)
    return torch.from_numpy(mel.T).float()          # (T, 80)


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_conformer(model, wav_np: np.ndarray, device: str):
    wav_t   = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_np.shape[0]], device=device)
    log_probs, enc_len, _ = model(input_signal=wav_t, input_signal_length=wav_len)
    T = int(enc_len[0].item())
    return log_probs[0, :T].cpu().float().numpy()   # (T, V)


@torch.no_grad()
def run_soft_aligner(model, wav_np: np.ndarray, device: str):
    mel = get_mel(wav_np).to(device)               # (T, 80)
    mel_t   = mel.unsqueeze(0)
    mel_len = torch.tensor([mel.shape[0]], device=device)
    log_probs, out_len = model(mel_t, mel_len)
    T2 = int(out_len[0].item())
    return log_probs[0, :T2].cpu().float().numpy()  # (T', V)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_utterance(wav_np, row, conformer_lp, soft_lp,
                   blank_id, tokenizer, ax_mel, ax_conf, ax_soft):
    tokens = tokenizer.text_to_ids(row["text"])
    labels = [tokenizer.ids_to_text([t]) for t in tokens]

    # --- Forced alignments ---
    conf_align = viterbi_align(conformer_lp, tokens, blank_id)
    soft_align = viterbi_align(soft_lp,      tokens, blank_id)

    # --- Mel (using conformer frame count as reference) ---
    T_conf = conformer_lp.shape[0]
    dur_sec = wav_np.shape[0] / 16000.0
    time_conf = np.linspace(0, dur_sec, T_conf)
    time_soft = np.linspace(0, dur_sec, soft_lp.shape[0])

    # mel spectrogram (rough, from raw audio via librosa)
    try:
        import librosa
        mel_s = librosa.feature.melspectrogram(y=wav_np, sr=16000, n_mels=80,
                                                hop_length=160, n_fft=512)
        mel_s = librosa.power_to_db(mel_s + 1e-6)
        ax_mel.imshow(mel_s, origin="lower", aspect="auto",
                      extent=[0, dur_sec, 0, 80], cmap="magma")
    except Exception:
        ax_mel.plot(np.linspace(0, dur_sec, len(wav_np)), wav_np)
    ax_mel.set_title(f'"{row["text"]}"', fontsize=8)
    ax_mel.set_ylabel("mel", fontsize=7)
    ax_mel.set_xticks([])

    def plot_posterior(ax, lp, align, time_axis, title):
        """
        Show per-token posterior (blank-excluded) for transcript tokens.
        Y-axis: each transcript token (bottom=first)
        X-axis: time
        Color: p_t(token k) (NOT blank-normalised — raw prob)
        """
        T, V = lp.shape
        probs = np.exp(lp)           # (T, V)
        # Gather only transcript tokens (no blank row)
        N = len(tokens)
        heat = np.zeros((N, T))
        for i, tok in enumerate(tokens):
            heat[i] = probs[:, tok]

        im = ax.imshow(heat, origin="lower", aspect="auto",
                       extent=[0, time_axis[-1], -0.5, N - 0.5],
                       cmap="hot", vmin=0, vmax=0.6, interpolation="nearest")
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time (s)", fontsize=7)

        # Viterbi path: mark non-blank frames
        for t, frame_label in enumerate(align):
            if frame_label != blank_id:
                tok_idx = tokens.index(frame_label) if frame_label in tokens else -1
                if tok_idx >= 0:
                    x = time_axis[t]
                    ax.plot(x, tok_idx, "c.", markersize=1.5, alpha=0.7)

        # Average token duration annotation
        durations = []
        prev = None; start = 0
        for t, lbl in enumerate(align):
            if lbl != blank_id and lbl != prev:
                if prev is not None and prev != blank_id:
                    durations.append(t - start)
                start = t
            prev = lbl
        avg_dur_ms = np.mean(durations) * 10 if durations else 0   # 10ms per frame
        ax.set_ylabel(f"transcript tokens\navg_dur={avg_dur_ms:.0f}ms", fontsize=7)

        return im

    plot_posterior(ax_conf, conformer_lp, conf_align, time_conf,
                   f"Conformer (peaky CTC)")
    plot_posterior(ax_soft, soft_lp,      soft_align, time_soft,
                   f"TDNN soft aligner (label prior α)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-ckpt",  required=True,
                    help="path to soft aligner checkpoint (best.pt)")
    ap.add_argument("--teacher",    default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer",  default="tokenizer_1024")
    ap.add_argument("--manifest",   default="data/dev_clean.json")
    ap.add_argument("--n",          type=int, default=4,
                    help="number of utterances to visualise")
    ap.add_argument("--min-dur",    type=float, default=3.0)
    ap.add_argument("--max-dur",    type=float, default=8.0)
    ap.add_argument("--out",        default="plots/soft_alignment.png")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load tokenizer
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    tokenizer = SentencePieceTokenizer(
        model_path=os.path.join(args.tokenizer, "tokenizer.model")
    )
    blank_id = tokenizer.vocab_size  # 1024

    # Load models
    print("[load] Conformer teacher ...", flush=True)
    conformer = load_conformer(args.teacher, args.device)

    print(f"[load] Soft aligner {args.soft_ckpt} ...", flush=True)
    soft, log_priors = load_soft_aligner(args.soft_ckpt, args.device)
    alpha = torch.load(args.soft_ckpt, map_location="cpu", weights_only=False)["args"].get("alpha", 0.3)
    p_blank = log_priors[blank_id].exp().item()
    print(f"  soft aligner: α={alpha}, P(blank) from training={p_blank:.3f}")

    # Select utterances
    rows = []
    with open(args.manifest) as f:
        for line in f:
            row = json.loads(line.strip())
            dur = row.get("duration", 0)
            if args.min_dur <= dur <= args.max_dur:
                rows.append(row)
    rows = rows[:args.n]
    print(f"[vis] {len(rows)} utterances")

    # Plot
    fig, axes = plt.subplots(3, len(rows),
                             figsize=(5 * len(rows), 9),
                             gridspec_kw={"height_ratios": [1, 2, 2]})
    if len(rows) == 1:
        axes = axes.reshape(-1, 1)

    for col, row in enumerate(rows):
        wav_np, sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(1)

        conf_lp = run_conformer(conformer, wav_np, args.device)
        soft_lp = run_soft_aligner(soft, wav_np, args.device)

        plot_utterance(wav_np, row, conf_lp, soft_lp, blank_id, tokenizer,
                       ax_mel=axes[0, col],
                       ax_conf=axes[1, col],
                       ax_soft=axes[2, col])

    fig.suptitle(
        "Peaky (Conformer) vs Soft (TDNN + label prior) alignment\n"
        "Color = p_t(token k)   Cyan dots = Viterbi path",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
