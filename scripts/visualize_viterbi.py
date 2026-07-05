#!/usr/bin/env python3
"""
Visualize CTC Viterbi forced alignment from teacher model.
Shows per-frame token assignments and segment statistics.
"""
import json
import sys
import os
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BLANK = 0
FLOOR = -1e9


def viterbi_forced_align(log_probs, tokens):
    """
    CTC forced alignment via Viterbi.
    log_probs: (T, K)  log-softmax
    tokens: list of token ids (no blanks, no repeats)
    Returns: (T,) array of token id per frame (blank=0)
    """
    # extended sequence: blank t1 blank t2 blank ... tN blank
    ext = []
    for t in tokens:
        ext.append(BLANK)
        ext.append(t)
    ext.append(BLANK)
    S = len(ext)
    T = log_probs.shape[0]

    log_probs = log_probs.float().cpu().numpy()

    # alpha[t, s] = max log-prob path ending at state s, frame t
    alpha = np.full((T, S), FLOOR)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    # backpointer
    bp = np.zeros((T, S), dtype=np.int32)

    for t in range(1, T):
        for s in range(S):
            # can stay, or come from s-1
            best_prev = alpha[t - 1, s]
            best_src = s
            if s > 0 and alpha[t - 1, s - 1] > best_prev:
                best_prev = alpha[t - 1, s - 1]
                best_src = s - 1
            # can skip blank: come from s-2 if ext[s] != blank and ext[s] != ext[s-2]
            if s > 1 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                if alpha[t - 1, s - 2] > best_prev:
                    best_prev = alpha[t - 1, s - 2]
                    best_src = s - 2
            alpha[t, s] = best_prev + log_probs[t, ext[s]]
            bp[t, s] = best_src

    # traceback from last valid state
    path_s = np.zeros(T, dtype=np.int32)
    s = S - 1 if alpha[T - 1, S - 1] > alpha[T - 1, S - 2] else S - 2
    path_s[T - 1] = s
    for t in range(T - 2, -1, -1):
        s = bp[t + 1, s]
        path_s[t] = s

    # map state index back to token (even states = blank, odd = token)
    frame_tokens = np.array([ext[s] for s in path_s], dtype=np.int32)
    return frame_tokens


def get_segments(frame_tokens, blank_id=0):
    """Extract (token, start, end) segments from per-frame alignment."""
    segments = []
    i = 0
    while i < len(frame_tokens):
        tok = frame_tokens[i]
        j = i
        while j < len(frame_tokens) and frame_tokens[j] == tok:
            j += 1
        segments.append((int(tok), i, j - 1, j - i))  # token, start, end, length
        i = j
    return segments


def main():
    import nemo.collections.asr as nemo_asr
    from torch.nn.utils.rnn import pad_sequence

    manifest_path = "data/train_clean_100.small_teacher.transition.json"
    n_samples = 8

    print("Loading teacher...")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_small")
    teacher = teacher.cuda().eval()
    tokenizer = teacher.tokenizer

    rows = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[:n_samples]

    print()
    seg_lengths_nonblank = []
    seg_lengths_blank = []

    for i, row in enumerate(rows):
        wav, sr = sf.read(row["audio_filepath"], dtype="float32")
        wav_t = torch.from_numpy(wav).unsqueeze(0).cuda()
        wav_len = torch.tensor([len(wav)], device="cuda")

        with torch.no_grad():
            log_probs, enc_len, _ = teacher.forward(input_signal=wav_t, input_signal_length=wav_len)

        lp = log_probs[0, :enc_len[0]]  # (T, K)
        T = lp.shape[0]

        # tokenize ground truth
        tokens = tokenizer.text_to_ids(row["text"])

        frame_tokens = viterbi_forced_align(lp, tokens)
        segments = get_segments(frame_tokens)

        print(f"[{i}] \"{row['text'][:60]}\"")
        print(f"     T={T} frames,  tokens={len(tokens)},  trans_target_M={len(row['teacher_target'])}")
        print(f"     segments: ", end="")
        for tok, st, en, length in segments:
            label = "_" if tok == BLANK else tokenizer.ids_to_text([tok])
            print(f"{label}({length})", end=" ")
        print()

        # stats
        for tok, st, en, length in segments:
            if tok == BLANK:
                seg_lengths_blank.append(length)
            else:
                seg_lengths_nonblank.append(length)
        print()

    print("=" * 60)
    print("Non-blank segment lengths:")
    print(f"  mean={np.mean(seg_lengths_nonblank):.1f}  median={np.median(seg_lengths_nonblank):.0f}"
          f"  min={np.min(seg_lengths_nonblank)}  max={np.max(seg_lengths_nonblank)}")
    print(f"  p25={np.percentile(seg_lengths_nonblank,25):.0f}"
          f"  p75={np.percentile(seg_lengths_nonblank,75):.0f}")
    print()
    print("Blank segment lengths:")
    print(f"  mean={np.mean(seg_lengths_blank):.1f}  median={np.median(seg_lengths_blank):.0f}"
          f"  min={np.min(seg_lengths_blank)}  max={np.max(seg_lengths_blank)}")


if __name__ == "__main__":
    main()
