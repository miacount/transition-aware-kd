#!/usr/bin/env python3
"""
Visualize the proposed temporal-support KD coordinates.

This script does not train a student. It checks whether the two ingredients
needed for span-level KD agree:

  gamma_T(t, u): teacher constrained CTC F-B occupancy for the u-th BPE token
  a(t, u):       phoneme-aligner F-B occupancy mapped to the u-th BPE token

The aligner is used only as temporal support. The teacher remains the semantic
BPE distribution source.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phoneme_utils import load_lexicon, text_to_phones, words_from_text  # noqa: E402
from train_soft_aligner import SoftAligner, _wav_to_mel  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward, plot_mel  # noqa: E402


def read_rows(path, n, min_dur, max_dur):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dur = row.get("duration", 0)
            if min_dur <= dur <= max_dur:
                rows.append(row)
            if len(rows) >= n:
                break
    return rows


def load_audio(path, sample_rate=16000):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav


def load_phoneme_aligner(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SoftAligner(
        input_dim=80,
        hidden=ckpt["args"].get("hidden", 512),
        vocab=len(ckpt["phone_to_id"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def run_phoneme_aligner(model, wav_np, log_priors, alpha, device):
    mel = _wav_to_mel(wav_np, 16000).to(device)
    log_probs, out_len = model(mel.unsqueeze(0), torch.tensor([mel.shape[0]], device=device))
    frames = int(out_len[0].item())
    lp = log_probs[0, :frames].detach().cpu().float().numpy()
    return lp - alpha * log_priors.reshape(1, -1)


@torch.no_grad()
def run_teacher(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


def bpe_piece_to_text(piece):
    if piece.startswith("▁"):
        return piece[1:]
    if piece.startswith("##"):
        return piece[2:]
    return piece


def map_bpe_to_phone_spans(text, bpe_tokens, lexicon, span_mode="proportional"):
    """Map each BPE occurrence to a contiguous global phone occurrence span.

    Uses word and character spans as the bridge. This is intentionally simple:
    word-internal BPE character spans are projected to word-internal phoneme
    spans by relative position.
    """
    words = words_from_text(text)
    word_prons = []
    missing = []
    for w in words:
        pron = lexicon.get(w)
        if pron is None:
            missing.append(w)
            pron = []
        word_prons.append(pron)
    if not words:
        return None, None, missing

    phone_offsets = []
    offset = 0
    for pron in word_prons:
        phone_offsets.append(offset)
        offset += len(pron)

    spans = []
    word_idx = -1
    char_cursor = 0
    for tok in bpe_tokens:
        tok = str(tok)
        starts_word = tok.startswith("▁") or tok.startswith("##") is False and word_idx < 0
        if tok.startswith("▁") or starts_word:
            word_idx += 1
            char_cursor = 0
        if word_idx < 0 or word_idx >= len(words):
            return None, None, missing + ["<bpe_word_mismatch>"]

        pron = word_prons[word_idx]
        if not pron:
            # word missing from lexicon: zero-length span -> zero temporal
            # support downstream, rather than discarding the whole utterance.
            offs = phone_offsets[word_idx]
            spans.append((offs, offs))
            continue

        piece = bpe_piece_to_text(tok).lower().replace("'", "")
        word = words[word_idx].replace("'", "")

        if piece:
            found = word.find(piece, char_cursor)
            if found >= 0:
                c0 = found
                c1 = found + len(piece)
            else:
                c0 = char_cursor
                c1 = min(len(word), char_cursor + len(piece))
        else:
            c0 = char_cursor
            c1 = char_cursor
        c0 = max(0, min(c0, len(word)))
        c1 = max(c0, min(c1, len(word)))
        char_cursor = c1

        n_chars = max(len(word), 1)
        n_phones = len(pron)
        if span_mode == "word":
            p0, p1 = 0, n_phones
        else:
            p0 = int(round(c0 / n_chars * n_phones))
            p1 = int(round(c1 / n_chars * n_phones))
            if p1 <= p0:
                center = (c0 + c1) / 2.0 / n_chars
                p0 = int(np.floor(center * n_phones))
                p1 = p0 + 1
            p0 = max(0, min(p0, n_phones - 1))
            p1 = max(p0 + 1, min(p1, n_phones))
        spans.append((phone_offsets[word_idx] + p0, phone_offsets[word_idx] + p1))

    phone_seq = [p for pron in word_prons for p in pron]
    return spans, phone_seq, []


def resample_time(mat, target_frames):
    if mat.shape[0] == target_frames:
        return mat
    src_x = np.linspace(0.0, 1.0, mat.shape[0])
    dst_x = np.linspace(0.0, 1.0, target_frames)
    out = np.zeros((target_frames, mat.shape[1]), dtype=np.float64)
    for u in range(mat.shape[1]):
        out[:, u] = np.interp(dst_x, src_x, mat[:, u])
    return out


def normalize_cols(mat):
    return mat / np.maximum(mat.sum(axis=0, keepdims=True), 1e-12)


def bpe_labels(tokenizer, ids):
    return [t.replace("▁", "_") for t in tokenizer.ids_to_tokens([int(i) for i in ids])]


def plot_case(row, wav, teacher_lp, teacher_gamma, bpe_mask, dot_gates, coverage_gates,
              bpe_ids, tokenizer, sample_rate, axes):
    duration = len(wav) / float(sample_rate)
    labels = bpe_labels(tokenizer, bpe_ids)
    time = np.linspace(0.0, duration, teacher_gamma.shape[0])
    plot_mel(axes[0], wav, sample_rate, duration)
    axes[0].set_title(f'"{row["text"]}"', fontsize=8)

    def heat(ax, mat, title, cmap):
        h = mat.T
        ax.imshow(
            h,
            origin="lower",
            aspect="auto",
            extent=[0, duration, -0.5, len(bpe_ids) - 0.5],
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.2, float(np.quantile(h, 0.995))),
        )
        ax.set_ylabel(title, fontsize=7)
        ax.set_xticks([])
        if len(labels) <= 45:
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels, fontsize=5)
        else:
            ax.set_yticks([])

    heat(axes[1], teacher_gamma, "teacher gamma_T", "hot")
    heat(axes[2], bpe_mask, "aligner support a", "viridis")

    diff = np.minimum(normalize_cols(teacher_gamma), normalize_cols(bpe_mask))
    heat(axes[3], diff, "overlap min", "magma")
    axes[3].set_xlabel("time (s)", fontsize=7)

    ax = axes[4]
    x = np.arange(len(dot_gates))
    ax.bar(x - 0.18, dot_gates, color="#4477AA", width=0.36, label="dot")
    ax.bar(x + 0.18, coverage_gates, color="#CC6677", width=0.36, label="coverage")
    ax.set_ylim(0, max(0.1, float(max(np.max(dot_gates), np.max(coverage_gates))) * 1.15))
    ax.set_ylabel("gate", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("BPE occurrence u", fontsize=7)
    ax.grid(alpha=0.2, axis="y")
    if len(labels) <= 45:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)

    probs = np.exp(teacher_lp)
    q_top_match = []
    for u, tok in enumerate(bpe_ids):
        q = (teacher_gamma[:, u:u + 1] * probs).sum(axis=0)
        q[int(tok)] = max(q[int(tok)], 0.0)
        q[-1] = 0.0
        top = int(q.argmax())
        q_top_match.append(top == int(tok))
    axes[4].text(
        0.01,
        0.92,
        f"dot={np.mean(dot_gates):.3f}/{np.quantile(dot_gates, 0.1):.3f}  cov={np.mean(coverage_gates):.3f}/{np.quantile(coverage_gates, 0.1):.3f}  teacher top1={100*np.mean(q_top_match):.1f}%",
        transform=axes[4].transAxes,
        fontsize=7,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-ckpt", default="nemo_experiments/phoneme_soft_aligner/epoch_010.pt")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=8.0)
    ap.add_argument("--out", default="analysis/temporal_support_kd/dev_clean_support.png")
    ap.add_argument("--span-mode", choices=["proportional", "word"], default="proportional",
                    help="how to map BPE pieces inside a word to phoneme occurrences")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    bpe_blank = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)

    phone_model, phone_ckpt = load_phoneme_aligner(args.phoneme_ckpt, args.device)
    phone_to_id = phone_ckpt["phone_to_id"]
    phone_blank = int(phone_ckpt["blank_id"])
    phone_alpha = float(phone_ckpt["args"].get("alpha", 0.3))
    phone_log_priors = phone_ckpt["log_priors"].float().numpy()
    lexicon = load_lexicon(phone_ckpt["args"].get("lexicon") or None)

    rows = read_rows(args.manifest, args.n * 4, args.min_dur, args.max_dur)
    cases = []
    summaries = []
    for row in rows:
        if len(cases) >= args.n:
            break
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != bpe_blank]
        bpe_toks = tokenizer.ids_to_tokens(bpe_ids)
        spans, phone_seq, missing = map_bpe_to_phone_spans(row["text"], bpe_toks, lexicon, span_mode=args.span_mode)
        if missing or not spans:
            continue
        phone_ids = [phone_to_id[p] for p in phone_seq]
        wav = load_audio(row["audio_filepath"], sample_rate)
        teacher_lp = run_teacher(teacher, wav, args.device)
        teacher_fb = ctc_forward_backward(teacher_lp, bpe_ids, bpe_blank)
        teacher_gamma = teacher_fb["token_gamma"]
        phone_scores = run_phoneme_aligner(phone_model, wav, phone_log_priors, phone_alpha, args.device)
        phone_fb = ctc_forward_backward(phone_scores, phone_ids, phone_blank)
        phone_occ = phone_fb["token_gamma"]

        bpe_mask_phone = np.zeros((phone_occ.shape[0], len(bpe_ids)), dtype=np.float64)
        for u, (s, e) in enumerate(spans):
            bpe_mask_phone[:, u] = phone_occ[:, s:e].sum(axis=1)
        bpe_mask = resample_time(bpe_mask_phone, teacher_gamma.shape[0])
        gt = normalize_cols(teacher_gamma)
        ga = normalize_cols(bpe_mask)
        dot_gates = (gt * ga).sum(axis=0)
        mask_peak = bpe_mask / np.maximum(bpe_mask.max(axis=0, keepdims=True), 1e-12)
        coverage_gates = (gt * mask_peak).sum(axis=0)
        cases.append((row, wav, teacher_lp, teacher_gamma, bpe_mask, dot_gates, coverage_gates, bpe_ids))
        summaries.append({
            "audio_filepath": row["audio_filepath"],
            "text": row["text"],
            "tokens": len(bpe_ids),
            "phones": len(phone_ids),
            "dot_gate_mean": float(dot_gates.mean()),
            "dot_gate_median": float(np.median(dot_gates)),
            "dot_gate_p10": float(np.quantile(dot_gates, 0.1)),
            "coverage_gate_mean": float(coverage_gates.mean()),
            "coverage_gate_median": float(np.median(coverage_gates)),
            "coverage_gate_p10": float(np.quantile(coverage_gates, 0.1)),
            "span_mode": args.span_mode,
        })

    if not cases:
        raise ValueError("No usable utterances; check lexicon/token mapping coverage.")

    fig, axes = plt.subplots(
        len(cases) * 5,
        1,
        figsize=(16, 4.4 * len(cases)),
        gridspec_kw={"height_ratios": [0.8, 1.4, 1.4, 1.4, 0.8] * len(cases), "hspace": 0.35},
    )
    if len(cases) == 1:
        axes = np.asarray(axes)
    for i, case in enumerate(cases):
        plot_case(*case, tokenizer=tokenizer, sample_rate=sample_rate, axes=axes[i * 5:(i + 1) * 5])

    fig.suptitle("Temporal-Support KD Diagnostic: teacher gamma vs phoneme-derived BPE support", fontsize=11, y=0.995)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = Path(args.out).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
