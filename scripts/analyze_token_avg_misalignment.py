#!/usr/bin/env python3
"""
Measure token_avg KD misalignment: at teacher Viterbi peak frames,
what is the trained student actually outputting?

For each non-blank segment k in teacher Viterbi alignment:
  - t_T = actual teacher peak frame: argmax p_teacher(token_k) within [seg_start, seg_end)
  - student argmax at EXACTLY t_T → correct? blank?
  - t_S = student peak frame: argmax p_student(token_k) near [seg_start, seg_end)
  - offset = |t_T - t_S|
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel
import nemo.collections.asr as nemo_asr


def load_audio(path, sr=16000):
    wav, orig_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return torch.from_numpy(wav)


def load_student(config_path, ckpt_path, device):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if model_cfg.get("test_ds") is not None:
            del model_cfg.test_ds
    model = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def get_logprobs(model, wav, device):
    wav_t = wav.unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    T = int(enc_len[0].item())
    return log_probs[0, :T].float().cpu()  # (T, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  default="data/train_clean_100.small_teacher.token_avg.json")
    ap.add_argument("--data_dir",  default="data",
                    help="prefix for relative teacher_token_avg_path entries")
    ap.add_argument("--config",    default="configs/student_base.yaml")
    ap.add_argument("--ckpt",      required=True, help="trained student checkpoint")
    ap.add_argument("--teacher",   default="stt_en_conformer_ctc_small")
    ap.add_argument("--limit",     type=int, default=300)
    ap.add_argument("--search_radius", type=int, default=5,
                    help="frames to search beyond segment boundary for student peak")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"[load] teacher: {args.teacher}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(device).eval()
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)

    print(f"[load] student: {args.ckpt}")
    student = load_student(args.config, args.ckpt, device)

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    print(f"[analyze] {len(rows)} utterances  blank_id={blank_id}")

    total_segs = 0
    correct_at_t_T = 0        # student argmax == token_k at exactly t_T
    blank_at_t_T = 0          # student outputs blank at t_T
    offset_list = []          # |t_T - t_S|
    seg_len_buckets = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in rows:
        wav = load_audio(row["audio_filepath"])

        pt_path = row["teacher_token_avg_path"]
        if not os.path.isabs(pt_path):
            pt_path = os.path.join(args.data_dir, pt_path)
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)

        seg_starts = pt["seg_starts"].tolist()
        seg_ends   = pt["seg_ends"].tolist()
        avg_ids    = pt["avg_ids"]       # (N, 8) top-k token ids
        T_teacher  = int(pt["teacher_frames"])

        # actual teacher logprobs (real inference)
        t_logprobs = get_logprobs(teacher, wav, device)  # (T_teacher, V)
        s_logprobs = get_logprobs(student, wav, device)  # (T_student, V)

        T_t = t_logprobs.shape[0]
        T_s = s_logprobs.shape[0]
        scale = T_s / T_t if T_t > 0 else 1.0

        for k in range(len(seg_starts)):
            tok_id = int(avg_ids[k, 0].item())
            if tok_id == blank_id:
                continue

            t_start = int(seg_starts[k])
            t_end   = int(seg_ends[k])
            seg_len = t_end - t_start

            t_start_c = min(t_start, T_t - 1)
            t_end_c   = min(t_end,   T_t)

            # --- actual teacher peak frame ---
            teacher_seg_probs = t_logprobs[t_start_c:t_end_c, tok_id]  # (seg_len,)
            t_T = t_start_c + int(teacher_seg_probs.argmax().item())

            # map t_T to student frame space
            s_T = min(int(round(t_T * scale)), T_s - 1)

            # student argmax at exactly s_T
            s_argmax = int(s_logprobs[s_T].argmax().item())
            correct  = (s_argmax == tok_id)
            is_blank = (s_argmax == blank_id)

            correct_at_t_T += int(correct)
            blank_at_t_T   += int(is_blank)

            # student peak frame for this token (search around mapped segment)
            s_seg_start = max(0, min(int(t_start_c * scale), T_s - 1) - args.search_radius)
            s_seg_end   = min(T_s, min(int(t_end_c * scale), T_s) + args.search_radius)
            if s_seg_end > s_seg_start:
                s_window = s_logprobs[s_seg_start:s_seg_end, tok_id]
                t_S = s_seg_start + int(s_window.argmax().item())
                offset_list.append(abs(s_T - t_S))

            total_segs += 1
            seg_len_buckets[min(seg_len, 5)]["total"]   += 1
            seg_len_buckets[min(seg_len, 5)]["correct"] += int(correct)

    if total_segs == 0:
        print("no segments found")
        return

    print(f"\n=== Token-avg KD Misalignment (teacher-inferred peaks) ===")
    print(f"segments analyzed                 : {total_segs}")
    print(f"student correct at t_T (exact)    : {correct_at_t_T}/{total_segs} "
          f"({100*correct_at_t_T/total_segs:.1f}%)")
    print(f"student blank at t_T              : {blank_at_t_T}/{total_segs} "
          f"({100*blank_at_t_T/total_segs:.1f}%)")

    if offset_list:
        arr = np.array(offset_list)
        print(f"\n--- student peak offset |t_S - t_T| (search ±{args.search_radius} beyond segment) ---")
        print(f"mean offset   : {arr.mean():.2f} frames")
        print(f"median offset : {np.median(arr):.1f} frames")
        print(f"max offset    : {arr.max()} frames")
        for thr in [0, 1, 2, 3, 5]:
            n = int((arr <= thr).sum())
            print(f"offset <= {thr}  : {n}/{len(arr)} ({100*n/len(arr):.1f}%)")

    print(f"\n--- correctness by teacher segment length ---")
    print(f"{'seg_len':>8} {'total':>8} {'correct':>8} {'acc%':>8}")
    for k in sorted(seg_len_buckets):
        b = seg_len_buckets[k]
        label = f"{k}+" if k == 5 else str(k)
        acc = 100 * b["correct"] / b["total"] if b["total"] else 0
        print(f"{label:>8} {b['total']:>8} {b['correct']:>8} {acc:>7.1f}%")


if __name__ == "__main__":
    main()
