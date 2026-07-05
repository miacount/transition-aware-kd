#!/usr/bin/env python3
"""
Visualize transcript-conditioned CTC forward-backward alignment.

This diagnoses whether Conformer teacher CTC can provide a useful soft
alignment signal before we build KD targets from it.

For each utterance, the figure shows:
  1. log-mel spectrogram
  2. teacher token posterior for transcript BPE positions, with Viterbi path
  3. CTC forward-backward occupancy gamma for transcript BPE positions
  4. blank occupancy and max token occupancy

Example:
  python scripts/visualize_ctc_forward_backward.py \
    --manifest data/dev_clean.json \
    --teacher stt_en_conformer_ctc_small \
    --n 4 \
    --out analysis/fb_alignment/dev_clean_fb.png
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch


NEG_INF = -1.0e30


def read_manifest(path, n=0, offset=0):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if offset:
        rows = rows[offset:]
    return rows[:n] if n else rows


def load_audio(path, sample_rate):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav


def make_ext_labels(tokens, blank_id):
    ext = []
    for tok in tokens:
        ext.append(blank_id)
        ext.append(int(tok))
    ext.append(blank_id)
    return np.asarray(ext, dtype=np.int64)


def logsumexp_np(values):
    values = np.asarray(values, dtype=np.float64)
    m = np.max(values)
    if m <= NEG_INF / 2:
        return NEG_INF
    return float(m + np.log(np.exp(values - m).sum()))


def ctc_forward_backward(log_probs, tokens, blank_id):
    """
    Args:
        log_probs: (T, V) log-probabilities.
        tokens: transcript token ids, no blanks.

    Returns:
        dict with ext labels, alpha, beta, state gamma, token gamma,
        blank gamma, and logZ.
    """
    ext = make_ext_labels(tokens, blank_id)
    T = log_probs.shape[0]
    S = len(ext)

    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    beta = np.full((T, S), NEG_INF, dtype=np.float64)

    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            prev = [alpha[t - 1, s]]
            if s > 0:
                prev.append(alpha[t - 1, s - 1])
            if s > 1 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                prev.append(alpha[t - 1, s - 2])
            alpha[t, s] = logsumexp_np(prev) + log_probs[t, ext[s]]

    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0

    for t in range(T - 2, -1, -1):
        for s in range(S):
            nxt = [beta[t + 1, s] + log_probs[t + 1, ext[s]]]
            if s + 1 < S:
                nxt.append(beta[t + 1, s + 1] + log_probs[t + 1, ext[s + 1]])
            if (
                s + 2 < S
                and ext[s + 2] != blank_id
                and ext[s + 2] != ext[s]
            ):
                nxt.append(beta[t + 1, s + 2] + log_probs[t + 1, ext[s + 2]])
            beta[t, s] = logsumexp_np(nxt)

    if S == 1:
        log_z = alpha[T - 1, 0]
    else:
        log_z = logsumexp_np([alpha[T - 1, S - 1], alpha[T - 1, S - 2]])

    gamma_state = np.exp(alpha + beta - log_z)
    gamma_state = np.nan_to_num(gamma_state, nan=0.0, posinf=0.0, neginf=0.0)

    token_gamma = gamma_state[:, 1::2]
    blank_gamma = gamma_state[:, 0::2].sum(axis=1)
    return {
        "ext": ext,
        "alpha": alpha,
        "beta": beta,
        "gamma_state": gamma_state,
        "token_gamma": token_gamma,
        "blank_gamma": blank_gamma,
        "log_z": float(log_z),
    }


def viterbi_states(log_probs, tokens, blank_id):
    ext = make_ext_labels(tokens, blank_id)
    T = log_probs.shape[0]
    S = len(ext)
    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int32)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

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
            alpha[t, s] = best + log_probs[t, ext[s]]
            bp[t, s] = src

    if S == 1:
        s = 0
    else:
        s = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2
    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = s
    for t in range(T - 2, -1, -1):
        s = bp[t + 1, s]
        states[t] = s
    return states, ext


def token_labels(tokenizer, tokens):
    labels = []
    for tok in tokens:
        try:
            labels.append(tokenizer.ids_to_tokens([int(tok)])[0].replace("▁", "_"))
        except Exception:
            labels.append(str(int(tok)))
    return labels


def plot_mel(ax, wav, sample_rate, duration):
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1.0e-6)
        ax.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            extent=[0, duration, 0, 80],
            cmap="magma",
        )
    except Exception:
        ax.plot(np.linspace(0, duration, len(wav)), wav, linewidth=0.5)
    ax.set_ylabel("mel", fontsize=7)
    ax.set_xticks([])


def plot_one(row, wav, log_probs, tokens, tokenizer, blank_id, sample_rate, axes):
    T = log_probs.shape[0]
    duration = len(wav) / float(sample_rate)
    time = np.linspace(0.0, duration, T)
    probs = np.exp(log_probs)

    fb = ctc_forward_backward(log_probs, tokens, blank_id)
    vit_states, _ = viterbi_states(log_probs, tokens, blank_id)
    vit_token_pos = np.full(T, -1, dtype=np.int32)
    odd = (vit_states % 2) == 1
    vit_token_pos[odd] = (vit_states[odd] - 1) // 2

    labels = token_labels(tokenizer, tokens)
    N = len(tokens)
    token_prob = np.zeros((N, T), dtype=np.float32)
    for i, tok in enumerate(tokens):
        token_prob[i] = probs[:, int(tok)]

    plot_mel(axes[0], wav, sample_rate, duration)
    axes[0].set_title(f'"{row["text"]}"', fontsize=8)

    vmax_prob = min(1.0, max(0.2, float(np.quantile(token_prob, 0.995))))
    axes[1].imshow(
        token_prob,
        origin="lower",
        aspect="auto",
        extent=[0, duration, -0.5, N - 0.5],
        cmap="hot",
        vmin=0.0,
        vmax=vmax_prob,
        interpolation="nearest",
    )
    for t, pos in enumerate(vit_token_pos):
        if pos >= 0:
            axes[1].plot(time[t], pos, ".", color="#42d9ff", markersize=1.6, alpha=0.8)
    axes[1].set_ylabel("teacher p(token)", fontsize=7)
    axes[1].set_xticks([])

    gamma = fb["token_gamma"].T
    vmax_gamma = min(1.0, max(0.2, float(np.quantile(gamma, 0.995))))
    axes[2].imshow(
        gamma,
        origin="lower",
        aspect="auto",
        extent=[0, duration, -0.5, N - 0.5],
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_gamma,
        interpolation="nearest",
    )
    axes[2].set_ylabel("FB gamma", fontsize=7)
    axes[2].set_xticks([])

    if N <= 45:
        yticks = np.arange(N)
        for ax in (axes[1], axes[2]):
            ax.set_yticks(yticks)
            ax.set_yticklabels(labels, fontsize=5)
    else:
        for ax in (axes[1], axes[2]):
            ax.set_yticks([])

    axes[3].plot(time, fb["blank_gamma"], label="FB blank occupancy", linewidth=1.0)
    axes[3].plot(
        time,
        fb["token_gamma"].max(axis=1),
        label="FB max token occupancy",
        linewidth=1.0,
    )
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_ylabel("occupancy", fontsize=7)
    axes[3].set_xlabel("time (s)", fontsize=7)
    axes[3].legend(fontsize=6, loc="upper right")

    hard_token_frames = int((vit_token_pos >= 0).sum())
    eff_dur = fb["token_gamma"].sum(axis=0)
    one_frame_like = int((eff_dur < 1.5).sum())
    summary = {
        "audio_filepath": row["audio_filepath"],
        "text": row["text"],
        "frames": int(T),
        "tokens": int(N),
        "viterbi_nonblank_frames": hard_token_frames,
        "viterbi_nonblank_frame_rate": hard_token_frames / max(T, 1),
        "fb_mean_blank_occupancy": float(fb["blank_gamma"].mean()),
        "fb_mean_effective_token_duration": float(eff_dur.mean()) if N else 0.0,
        "fb_median_effective_token_duration": float(np.median(eff_dur)) if N else 0.0,
        "fb_one_frame_like_tokens": one_frame_like,
        "fb_one_frame_like_token_rate": one_frame_like / max(N, 1),
        "ctc_log_z": fb["log_z"],
    }
    axes[3].text(
        0.01,
        0.05,
        (
            f"Viterbi nonblank={100*summary['viterbi_nonblank_frame_rate']:.1f}% | "
            f"FB blank={summary['fb_mean_blank_occupancy']:.2f} | "
            f"FB token dur median={summary['fb_median_effective_token_duration']:.2f}f | "
            f"<1.5f tokens={100*summary['fb_one_frame_like_token_rate']:.1f}%"
        ),
        transform=axes[3].transAxes,
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )
    return summary


@torch.no_grad()
def run_teacher(model, wav, device):
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(
        input_signal=wav_t,
        input_signal_length=wav_len,
    )
    T = int(enc_len[0].item())
    return log_probs[0, :T].detach().cpu().float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="analysis/fb_alignment/ctc_fb_alignment.png")
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    rows = read_manifest(args.manifest, n=args.n, offset=args.offset)
    if not rows:
        raise ValueError(f"no rows selected from {args.manifest}")

    teacher_name = (
        "stt_en_conformer_ctc_small"
        if args.teacher == "stt_en_conformer_small"
        else args.teacher
    )
    print(f"[load] {teacher_name}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(teacher_name)
    teacher = teacher.to(args.device).eval()
    teacher.freeze()
    tokenizer = teacher.tokenizer
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    print(f"[blank_id] {blank_id}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".json")

    fig, axes = plt.subplots(
        len(rows) * 4,
        1,
        figsize=(16, max(4.0, 3.2 * len(rows))),
        gridspec_kw={"height_ratios": [0.8, 1.3, 1.3, 0.8] * len(rows), "hspace": 0.35},
    )
    if len(rows) == 1:
        axes = np.asarray(axes)

    summaries = []
    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {row['audio_filepath']}")
        wav = load_audio(row["audio_filepath"], sample_rate)
        log_probs = run_teacher(teacher, wav, args.device)
        tokens = tokenizer.text_to_ids(row["text"])
        tokens = [int(t) for t in tokens if int(t) != blank_id]
        if not tokens:
            continue
        ax_group = axes[i * 4 : (i + 1) * 4]
        summaries.append(
            plot_one(row, wav, log_probs, tokens, tokenizer, blank_id, sample_rate, ax_group)
        )

    fig.suptitle(
        "CTC Forward-Backward Soft Alignment vs Viterbi Hard Alignment",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"[saved] figure : {out_path}")
    print(f"[saved] summary: {summary_path}")


if __name__ == "__main__":
    main()
