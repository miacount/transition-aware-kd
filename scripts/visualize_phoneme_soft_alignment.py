#!/usr/bin/env python3
"""
Visualize phoneme soft alignment beside Conformer BPE teacher posteriors.

This is the bridge diagnostic for the hybrid idea:
  phoneme aligner -> soft temporal masks
  Conformer BPE teacher -> BPE identity distributions

Example:
  python scripts/visualize_phoneme_soft_alignment.py \
    --phoneme-ckpt nemo_experiments/phoneme_soft_aligner/best.pt \
    --lexicon data/cmudict.dict \
    --manifest data/dev_clean.json \
    --n 4 \
    --out analysis/phoneme_soft_alignment/dev_clean.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phoneme_utils import load_lexicon, text_to_phone_ids  # noqa: E402
from train_soft_aligner import SoftAligner, _wav_to_mel  # noqa: E402


FLOOR = -1.0e9


def viterbi_align(log_probs_np, tokens, blank_id):
    ext = []
    for tok in tokens:
        ext.append(blank_id)
        ext.append(int(tok))
    ext.append(blank_id)
    T = log_probs_np.shape[0]
    S = len(ext)
    alpha = np.full((T, S), FLOOR, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int32)
    alpha[0, 0] = log_probs_np[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs_np[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, src = alpha[t - 1, s], s
            if s > 0 and alpha[t - 1, s - 1] > best:
                best, src = alpha[t - 1, s - 1], s - 1
            if (
                s > 1
                and ext[s] != blank_id
                and ext[s] != ext[s - 2]
                and alpha[t - 1, s - 2] > best
            ):
                best, src = alpha[t - 1, s - 2], s - 2
            alpha[t, s] = best + log_probs_np[t, ext[s]]
            bp[t, s] = src
    s = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2
    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = s
    for t in range(T - 2, -1, -1):
        s = bp[t + 1, s]
        states[t] = s
    labels = np.array([ext[s] for s in states], dtype=np.int32)
    positions = np.full(T, -1, dtype=np.int32)
    odd = (states % 2) == 1
    positions[odd] = (states[odd] - 1) // 2
    return labels, positions


def load_phoneme_aligner(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = SoftAligner(
        input_dim=80,
        hidden=args.get("hidden", 512),
        vocab=len(ckpt["phone_to_id"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def run_phoneme_aligner(model, wav_np, device):
    mel = _wav_to_mel(wav_np, 16000).to(device)
    mel_t = mel.unsqueeze(0)
    mel_len = torch.tensor([mel.shape[0]], device=device)
    log_probs, out_len = model(mel_t, mel_len)
    frames = int(out_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


@torch.no_grad()
def run_conformer(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


def plot_mel(ax, wav_np, duration):
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=wav_np,
            sr=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1e-6)
        ax.imshow(mel_db, origin="lower", aspect="auto", extent=[0, duration, 0, 80], cmap="magma")
    except Exception:
        ax.plot(np.linspace(0, duration, len(wav_np)), wav_np, linewidth=0.5)
    ax.set_ylabel("mel", fontsize=7)
    ax.set_xticks([])


def token_labels(tokenizer, tokens):
    labels = []
    for tok in tokens:
        labels.append(tokenizer.ids_to_tokens([int(tok)])[0].replace("▁", "_"))
    return labels


def plot_utterance(
    row,
    wav_np,
    bpe_lp,
    phone_lp,
    phone_log_priors,
    phone_alpha,
    bpe_tokens,
    phone_ids,
    tokenizer,
    id_to_phone,
    bpe_blank,
    phone_blank,
    axes,
):
    duration = wav_np.shape[0] / 16000.0
    plot_mel(axes[0], wav_np, duration)
    axes[0].set_title(f'"{row["text"]}"', fontsize=8)

    bpe_probs = np.exp(bpe_lp)
    bpe_time = np.linspace(0.0, duration, bpe_lp.shape[0])
    bpe_heat = np.zeros((len(bpe_tokens), bpe_lp.shape[0]), dtype=np.float32)
    for i, tok in enumerate(bpe_tokens):
        bpe_heat[i] = bpe_probs[:, int(tok)]
    _, bpe_pos = viterbi_align(bpe_lp, bpe_tokens, bpe_blank)

    axes[1].imshow(
        bpe_heat,
        origin="lower",
        aspect="auto",
        extent=[0, duration, -0.5, len(bpe_tokens) - 0.5],
        cmap="hot",
        vmin=0.0,
        vmax=min(1.0, max(0.2, float(np.quantile(bpe_heat, 0.995)))),
        interpolation="nearest",
    )
    for t, pos in enumerate(bpe_pos):
        if pos >= 0:
            axes[1].plot(bpe_time[t], pos, ".", color="#42d9ff", markersize=1.5, alpha=0.8)
    axes[1].set_ylabel("Conformer BPE", fontsize=7)
    axes[1].set_xticks([])

    phone_scores = phone_lp - phone_alpha * phone_log_priors.reshape(1, -1)
    phone_scores_norm = phone_scores - phone_scores.max(axis=1, keepdims=True)
    phone_probs = np.exp(phone_scores_norm)
    phone_probs = phone_probs / np.maximum(phone_probs.sum(axis=1, keepdims=True), 1e-12)
    raw_phone_probs = np.exp(phone_lp)
    phone_time = np.linspace(0.0, duration, phone_lp.shape[0])
    phone_heat = np.zeros((len(phone_ids), phone_lp.shape[0]), dtype=np.float32)
    for i, pid in enumerate(phone_ids):
        phone_heat[i] = phone_probs[:, int(pid)]
    _, phone_pos = viterbi_align(phone_scores, phone_ids, phone_blank)

    axes[2].imshow(
        phone_heat,
        origin="lower",
        aspect="auto",
        extent=[0, duration, -0.5, len(phone_ids) - 0.5],
        cmap="viridis",
        vmin=0.0,
        vmax=min(1.0, max(0.2, float(np.quantile(phone_heat, 0.995)))),
        interpolation="nearest",
    )
    for t, pos in enumerate(phone_pos):
        if pos >= 0:
            axes[2].plot(phone_time[t], pos, ".", color="#ffdf5d", markersize=1.5, alpha=0.8)
    axes[2].set_ylabel("phoneme aligner", fontsize=7)
    axes[2].set_xlabel("time (s)", fontsize=7)

    if len(bpe_tokens) <= 45:
        axes[1].set_yticks(np.arange(len(bpe_tokens)))
        axes[1].set_yticklabels(token_labels(tokenizer, bpe_tokens), fontsize=5)
    else:
        axes[1].set_yticks([])

    if len(phone_ids) <= 70:
        axes[2].set_yticks(np.arange(len(phone_ids)))
        axes[2].set_yticklabels([id_to_phone[int(i)] for i in phone_ids], fontsize=5)
    else:
        axes[2].set_yticks([])

    phone_blank_rate = float(phone_probs[:, phone_blank].mean())
    raw_phone_blank_rate = float(raw_phone_probs[:, phone_blank].mean())
    phone_hard_rate = float((phone_pos >= 0).mean())
    axes[2].text(
        0.01,
        0.04,
        f"prior P(blank)={phone_blank_rate:.2f} raw={raw_phone_blank_rate:.2f} | Viterbi phone frames={100*phone_hard_rate:.1f}%",
        transform=axes[2].transAxes,
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-ckpt", required=True)
    ap.add_argument("--lexicon", default="", help="same lexicon used for training")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=8.0)
    ap.add_argument("--out", default="analysis/phoneme_soft_alignment/phoneme_vs_bpe.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    import nemo.collections.asr as nemo_asr

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[load] phoneme aligner {args.phoneme_ckpt}")
    phone_model, ckpt = load_phoneme_aligner(args.phoneme_ckpt, args.device)
    phone_to_id = ckpt["phone_to_id"]
    id_to_phone = {int(k): v for k, v in ckpt["id_to_phone"].items()}
    phone_blank = int(ckpt["blank_id"])
    phone_log_priors = ckpt["log_priors"].float().numpy()
    phone_alpha = float(ckpt["args"].get("alpha", 0.3))
    lexicon = load_lexicon(args.lexicon or ckpt["args"].get("lexicon") or None)

    print(f"[load] teacher {args.teacher}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    bpe_blank = int(teacher.decoder.num_classes_with_blank - 1)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dur = row.get("duration", 0)
            if not (args.min_dur <= dur <= args.max_dur):
                continue
            phone_ids, _ = text_to_phone_ids(row["text"], lexicon, phone_to_id, unk_policy="skip")
            if phone_ids:
                row["_phone_ids"] = phone_ids
                rows.append(row)
            if len(rows) >= args.n:
                break
    print(f"[vis] {len(rows)} utterances")
    if not rows:
        raise ValueError("no visualizable rows; check lexicon coverage")

    fig, axes = plt.subplots(
        3,
        len(rows),
        figsize=(5.5 * len(rows), 9),
        gridspec_kw={"height_ratios": [1, 2, 2]},
    )
    if len(rows) == 1:
        axes = axes.reshape(3, 1)

    for col, row in enumerate(rows):
        wav_np, sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(1)
        if sr != 16000:
            import librosa
            wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)

        bpe_lp = run_conformer(teacher, wav_np, args.device)
        phone_lp = run_phoneme_aligner(phone_model, wav_np, args.device)
        bpe_tokens = [int(t) for t in tokenizer.text_to_ids(row["text"]) if int(t) != bpe_blank]
        plot_utterance(
            row,
            wav_np,
            bpe_lp,
            phone_lp,
            phone_log_priors,
            phone_alpha,
            bpe_tokens,
            row["_phone_ids"],
            tokenizer,
            id_to_phone,
            bpe_blank,
            phone_blank,
            axes[:, col],
        )

    fig.suptitle(
        "Conformer BPE Teacher vs Phoneme Label-Prior Soft Aligner",
        fontsize=11,
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
