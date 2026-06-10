#!/usr/bin/env python
"""Build teacher KD manifests.

Modes:
  transition : add teacher_target from teacher greedy CTC path collapse
  frame_topk : add teacher_frame_kd_path with per-frame top-k posterior .pt files
"""
import argparse
import json
from pathlib import Path

import torch


def read_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collapse_repeats(ids):
    out = []
    prev = None
    for x in ids:
        if x != prev:
            out.append(int(x))
            prev = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["transition", "frame_topk"], required=True)
    ap.add_argument("--manifest_in", required=True)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--out_dir", default="data/frame_kd")
    args = ap.parse_args()

    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    from torch.nn.utils.rnn import pad_sequence

    teacher_name = "stt_en_conformer_ctc_small" if args.teacher == "stt_en_conformer_small" else args.teacher
    print(f"[load] {teacher_name}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(teacher_name)
    teacher = teacher.to(args.device).eval()
    teacher.freeze()
    sample_rate = teacher.cfg.preprocessor.sample_rate

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    manifest_dir = Path(args.manifest_out).parent
    if args.mode == "frame_topk":
        (manifest_dir / out_dir.name).mkdir(parents=True, exist_ok=True)

    def load_audio(path):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        return torch.from_numpy(wav)

    t_lengths = []
    target_lengths = []
    topk_masses = []

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            sigs = [load_audio(row["audio_filepath"]) for row in batch]
            lens = torch.tensor([x.numel() for x in sigs], dtype=torch.long, device=args.device)
            padded = pad_sequence(sigs, batch_first=True).to(args.device)
            log_probs, enc_len, greedy = teacher.forward(input_signal=padded, input_signal_length=lens)
            greedy = greedy.detach().cpu()
            enc_len_cpu = enc_len.detach().cpu()

            for i, row in enumerate(batch):
                global_idx = start + i
                frames = int(enc_len_cpu[i].item())
                t_lengths.append(frames)

                if args.mode == "transition":
                    target = collapse_repeats(greedy[i, :frames].tolist())
                    row["teacher_target"] = target
                    target_lengths.append(len(target))
                else:
                    lp = log_probs[i, :frames].float()
                    if args.temperature != 1.0:
                        lp = torch.log_softmax(lp / args.temperature, dim=-1)
                    vals, ids = torch.topk(lp, k=min(args.top_k, lp.shape[-1]), dim=-1)
                    probs = vals.exp()
                    topk_masses.append(float(probs.sum(dim=-1).mean().detach().cpu()))
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    rel = Path(out_dir.name) / f"{global_idx:06d}.pt"
                    torch.save(
                        {
                            "ids": ids.detach().cpu().to(torch.int32),
                            "probs": probs.detach().cpu().to(torch.float16),
                            "frames": frames,
                            "top_k": int(ids.shape[-1]),
                            "temperature": float(args.temperature),
                        },
                        manifest_dir / rel,
                    )
                    row["teacher_frame_kd_path"] = str(rel)

            print(f"[prog] {min(start + args.batch_size, len(rows))}/{len(rows)}")

    write_manifest(args.manifest_out, rows)
    print("\n========== summary ==========")
    print(f"mode             : {args.mode}")
    print(f"utterances       : {len(rows)}")
    print(f"teacher T min/max: {min(t_lengths)}/{max(t_lengths)}")
    if target_lengths:
        print(f"target M min/max : {min(target_lengths)}/{max(target_lengths)}")
    if topk_masses:
        print(f"top-k/temp       : {args.top_k}/{args.temperature}")
        print(f"top-k mass mean  : {sum(topk_masses) / len(topk_masses):.4f}")
    print(f"written          : {args.manifest_out}")


if __name__ == "__main__":
    main()
