#!/usr/bin/env python3
"""
Analysis 3: Does blank-normalized posterior in blank frames carry token-relevant info?

For each blank frame near a Viterbi token spike, compute:
  r_t(k) = p_t(k) / (1 - p_t(blank))   for k != blank

Check if argmax(r_t) matches the neighboring token,
and how this hit rate decays with distance from the spike.

This validates whether Token Context Distillation is viable.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FLOOR = -1e9
MAX_DIST = 8


def viterbi_forced_align(log_probs_np, tokens, blank_id):
    ext = []
    for t in tokens:
        ext.append(blank_id)
        ext.append(t)
    ext.append(blank_id)
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
            if s > 1 and ext[s] != blank_id and ext[s] != ext[s - 2]:
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


def collapse_repeats(seq):
    out, prev = [], None
    for x in seq:
        if x != prev:
            out.append(int(x))
            prev = x
    return out


def get_spike_positions(frame_tokens, blank_id):
    spikes = []
    i = 0
    while i < len(frame_tokens):
        tok = int(frame_tokens[i])
        j = i
        while j < len(frame_tokens) and int(frame_tokens[j]) == tok:
            j += 1
        if tok != blank_id:
            spikes.append((tok, i, j))
        i = j
    return spikes


def analyze_utterance(prob, frame_tokens, T, blank_id):
    blank_prob = prob[:, blank_id]
    nonblank_prob = prob.copy()
    nonblank_prob[:, blank_id] = 0.0
    denom = nonblank_prob.sum(axis=1, keepdims=True).clip(min=1e-9)
    r = nonblank_prob / denom                        # blank-normalized (T, V)

    spikes = get_spike_positions(frame_tokens, blank_id)

    # Exclusive attribution: for each blank frame find the single nearest spike
    # (by distance to nearest edge of the spike).  Ties → skip (equidistant blanks
    # between two spikes carry mixed signal and would bias either direction).
    # direction: "right" = blank is to the RIGHT of the attributed spike
    #            "left"  = blank is to the LEFT  of the attributed spike
    blank_owner = {}  # t → (tok, dist, direction)
    for tok, sp_start, sp_end in spikes:
        for t in range(max(0, sp_start - MAX_DIST), min(T, sp_end + MAX_DIST)):
            if int(frame_tokens[t]) != blank_id:
                continue
            if t < sp_start:
                d = sp_start - t          # blank is LEFT of this spike
                direction = "left"
            else:                          # t >= sp_end
                d = t - (sp_end - 1)      # blank is RIGHT of this spike
                direction = "right"
            if d > MAX_DIST:
                continue

            prev = blank_owner.get(t)
            if prev is None:
                blank_owner[t] = (tok, d, direction)
            elif d < prev[1]:             # strictly closer → take over
                blank_owner[t] = (tok, d, direction)
            elif d == prev[1]:            # tie → mark as contested (skip)
                blank_owner[t] = None

    # Accumulate per (dist, direction) and per token (d=1 only)
    by_dist_dir = defaultdict(list)   # key: (dist, direction) → list of match tuples
    by_token    = defaultdict(list)   # key: tok_id (d=1 only) → list of (top1, top3)
    for t, owner in blank_owner.items():
        if owner is None:
            continue
        tok, dist, direction = owner
        r_t = r[t]
        bp  = float(blank_prob[t])
        top_indices = np.argpartition(r_t, -5)[-5:]
        top5 = top_indices[np.argsort(r_t[top_indices])[::-1]]

        top1_match = int(top5[0] == tok)
        top3_match = int(tok in top5[:3])
        top5_match = int(tok in top5)
        rank = int(np.where(np.argsort(r_t)[::-1] == tok)[0][0]) + 1

        by_dist_dir[(dist, direction)].append((bp, top1_match, top3_match, top5_match, rank))
        if dist == 1:
            by_token[tok].append((top1_match, top3_match))

    # Also return collapsed by_dist (unsigned, for backward-compat)
    by_dist = defaultdict(list)
    for (dist, _direction), vals in by_dist_dir.items():
        by_dist[dist].extend(vals)

    return by_dist, by_dist_dir, by_token


def load_student(ckpt, config, device):
    from omegaconf import OmegaConf, open_dict
    from model import TransitionKDModel
    cfg = OmegaConf.load(config)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if "test_ds" in model_cfg:
            del model_cfg.test_ds
    m = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(state.get("state_dict", state), strict=False)
    return m.to(device).eval()


def run_analysis(model, rows, blank_id, device, label, tokenizer=None):
    import soundfile as sf

    def load_audio(path):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return torch.from_numpy(wav)

    print(f"\n{'='*65}", flush=True)
    print(f"[{label}]  analyzing {len(rows)} utterances ...", flush=True)

    all_by_dist = defaultdict(list)
    all_by_dist_dir = defaultdict(list)
    all_by_token = defaultdict(list)
    total_frames = total_blank = total_nb = 0
    n_contested = 0

    with torch.no_grad():
        for row in rows:
            wav = load_audio(row["audio_filepath"]).unsqueeze(0).to(device)
            wav_len = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)
            log_probs, enc_len, greedy = model.forward(input_signal=wav, input_signal_length=wav_len)
            T = int(enc_len[0].item())
            lp = log_probs[0, :T].float()
            prob = lp.exp().cpu().numpy()
            lp_np = lp.cpu().numpy()
            greedy_path = greedy[0, :T].cpu().numpy()

            greedy_seq = collapse_repeats(greedy_path.tolist())
            tokens = [t for t in greedy_seq if t != blank_id]
            if not tokens:
                continue

            frame_tokens = viterbi_forced_align(lp_np, tokens, blank_id)
            by_dist, by_dist_dir, by_token = analyze_utterance(prob, frame_tokens, T, blank_id)
            for d, vals in by_dist.items():
                all_by_dist[d].extend(vals)
            for k, vals in by_dist_dir.items():
                all_by_dist_dir[k].extend(vals)
            for tok, vals in by_token.items():
                all_by_token[tok].extend(vals)

            total_frames += T
            total_blank  += int((frame_tokens == blank_id).sum())
            total_nb     += int((frame_tokens != blank_id).sum())

    print(f"utterances analyzed : {len(rows)}")
    print(f"total frames        : {total_frames}")
    print(f"blank frames        : {total_blank} ({100*total_blank/max(total_frames,1):.1f}%)")
    print(f"non-blank (spike)   : {total_nb} ({100*total_nb/max(total_frames,1):.1f}%)")
    print()
    print("blank-normalized posterior quality by distance from spike (exclusive attribution):")
    print(f"{'dist':>5}  {'n':>7}  {'blank_p':>8}  {'top1':>7}  {'top3':>7}  {'top5':>7}  {'med_rank':>9}")
    print("-" * 65)
    for dist in range(1, MAX_DIST + 1):
        vals = all_by_dist.get(dist, [])
        if not vals:
            continue
        print(f"  {dist:>3}  {len(vals):>7}  {np.mean([v[0] for v in vals]):>8.3f}"
              f"  {np.mean([v[1] for v in vals])*100:>6.1f}%"
              f"  {np.mean([v[2] for v in vals])*100:>6.1f}%"
              f"  {np.mean([v[3] for v in vals])*100:>6.1f}%"
              f"  {np.median([v[4] for v in vals]):>9.0f}")
    near = []
    for d in range(1, 4):
        near.extend(all_by_dist.get(d, []))
    if near:
        print(f"\nnear context (dist 1-3):  top1={np.mean([v[1] for v in near])*100:.1f}%  "
              f"top3={np.mean([v[2] for v in near])*100:.1f}%  "
              f"median_rank={np.median([v[4] for v in near]):.0f}")

    # Left / Right breakdown at d=1
    print()
    print("LEFT vs RIGHT breakdown at d=1 (exclusive attribution, ties excluded):")
    print(f"  {'dir':>6}  {'n':>7}  {'top1':>8}  {'top3':>8}")
    print("  " + "-" * 36)
    for direction in ("left", "right"):
        vals = all_by_dist_dir.get((1, direction), [])
        if not vals:
            print(f"  {direction:>6}  {'(empty)':>7}")
            continue
        print(f"  {direction:>6}  {len(vals):>7}  "
              f"{np.mean([v[1] for v in vals])*100:>7.1f}%  "
              f"{np.mean([v[2] for v in vals])*100:>7.1f}%")
    left1  = all_by_dist_dir.get((1, "left"),  [])
    right1 = all_by_dist_dir.get((1, "right"), [])
    if left1 and right1:
        lr = np.mean([v[1] for v in left1]) * 100
        rr = np.mean([v[1] for v in right1]) * 100
        print(f"\n  RIGHT − LEFT asymmetry (top1): {rr - lr:+.1f}pp")

    # Per-token breakdown at d=1 (top-25 by frequency)
    if all_by_token:
        print()
        print("Per-token blank context quality at d=1 (top-25 by count):")
        print(f"  {'token':<12}  {'n':>6}  {'top1':>7}  {'top3':>7}")
        print("  " + "-" * 38)
        rows_tok = sorted(all_by_token.items(), key=lambda x: -len(x[1]))[:25]
        for tok_id, vals in rows_tok:
            if tokenizer is not None:
                try:
                    tok_str = tokenizer.ids_to_text([tok_id])
                except Exception:
                    tok_str = f"id={tok_id}"
            else:
                tok_str = f"id={tok_id}"
            t1 = np.mean([v[0] for v in vals]) * 100
            t3 = np.mean([v[1] for v in vals]) * 100
            print(f"  {tok_str!r:<12}  {len(vals):>6}  {t1:>6.1f}%  {t3:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train_clean_100.small_teacher.transition.json")
    ap.add_argument("--teacher",  default="stt_en_conformer_ctc_small")
    ap.add_argument("--student-ckpt", default=None)
    ap.add_argument("--student-config", default="configs/student_base.yaml")
    ap.add_argument("--n",        type=int, default=200)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    print(f"[load] {args.teacher}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher)
    teacher = teacher.to(args.device).eval()
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    print(f"[blank_id] {blank_id}")

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[:args.n]
    print(f"analyzing {len(rows)} utterances ...")

    if not args.student_ckpt:
        run_analysis(teacher, rows, blank_id, args.device, "Teacher",
                     tokenizer=teacher.tokenizer)
    else:
        run_analysis(teacher, rows, blank_id, args.device, "Teacher",
                     tokenizer=teacher.tokenizer)
        del teacher
        torch.cuda.empty_cache()
        print(f"\n[load student] {args.student_ckpt}", flush=True)
        student = load_student(args.student_ckpt, args.student_config, args.device)
        print("[student loaded]", flush=True)
        run_analysis(student, rows, blank_id, args.device, "Student No-KD",
                     tokenizer=student.tokenizer)

    print()
    print("interpretation guide:")
    print("  top1 > 50%  → blank context carries token signal  → TCD viable")
    print("  top1 < 30%  → blank context is noisy              → TCD risky")


if __name__ == "__main__":
    main()
