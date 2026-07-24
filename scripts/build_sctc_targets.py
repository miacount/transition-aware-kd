#!/usr/bin/env python3
"""Build S-CTC (sequence-level KD) targets — Huang et al., Interspeech 2018.

For each utterance, run the teacher and compute the transcript-constrained CTC
forward-backward state-occupancy sigma'_CTC(k,t) (Eq. 4/9 in the paper). This is
the soft alignment of each ground-truth token (and blank) onto the teacher's
frames. The student is later trained by a per-frame cross-entropy against this
occupancy (Eq. 10):  L_S-CTC = -sum_t [ blank_occ(t)*log y_blank(t)
                                       + sum_u gamma(t,u)*log y_{id(u)}(t) ].

Stored per utterance (teacher_sctc_path):
  token_gamma : (N, T) fp16   per-frame occupancy of each transcript token
  bpe_ids     : (N,)   int32   vocab id of each transcript token
  teacher_frames : int         T (teacher frame count)

Unlike Span-KD this keeps the occupancy RAW (peaky, delta=0), keeps blank, and
carries no non-blank dark knowledge — it distills the teacher's alignment only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


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


def load_audio(path, sample_rate):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav


@torch.no_grad()
def run_teacher(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-in", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--out-dir", default="data/sctc")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="reuse existing per-utterance .pt targets")
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[:args.limit]

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    manifest_dir = Path(args.manifest_out).parent
    out_dir = Path(args.out_dir)
    (manifest_dir / out_dir.name).mkdir(parents=True, exist_ok=True)

    kept = 0
    reused = 0
    skipped = {"empty": 0}
    occ_means = []

    for idx, row in enumerate(rows):
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != blank]
        if not bpe_ids:
            skipped["empty"] += 1
            continue

        rel = Path(out_dir.name) / f"{idx:06d}.pt"
        out_path = manifest_dir / rel
        if args.resume and out_path.exists():
            try:
                torch.load(out_path, map_location="cpu", weights_only=False)
                row["teacher_sctc_path"] = str(rel)
                kept += 1
                reused += 1
                if kept % args.flush_every == 0:
                    write_manifest(args.manifest_out, rows)
                    print(f"[prog] kept={kept} reused={reused} processed={idx+1}/{len(rows)}", flush=True)
                continue
            except Exception as exc:
                print(f"[warn] failed to reuse {out_path}: {exc}; rebuilding", flush=True)

        wav = load_audio(row["audio_filepath"], sample_rate)
        teacher_lp = run_teacher(teacher, wav, args.device)              # (T, V)
        fb = ctc_forward_backward(teacher_lp, bpe_ids, blank)
        token_gamma = fb["token_gamma"]                                  # (T, N) transcript-constrained occupancy

        torch.save(
            {
                "token_gamma": torch.tensor(token_gamma.T, dtype=torch.float16),  # (N, T)
                "bpe_ids": torch.tensor(bpe_ids, dtype=torch.int32),              # (N,)
                "teacher_frames": int(token_gamma.shape[0]),
                "num_tokens": int(len(bpe_ids)),
                "mode": "sctc",
            },
            out_path,
        )
        row["teacher_sctc_path"] = str(rel)
        kept += 1
        occ_means.append(float(token_gamma.sum(axis=1).mean()))  # mean non-blank occupancy per frame
        if kept % args.flush_every == 0:
            write_manifest(args.manifest_out, rows)
            print(f"[prog] kept={kept} reused={reused} processed={idx+1}/{len(rows)}", flush=True)

    write_manifest(args.manifest_out, rows)
    print("\n========== summary ==========")
    print(f"input rows        : {len(rows)}")
    print(f"kept              : {kept}")
    print(f"reused            : {reused}")
    print(f"skipped           : {skipped}")
    if occ_means:
        print(f"mean non-blank occ/frame : {np.mean(occ_means):.4f}  (rest = blank)")
    print(f"written           : {args.manifest_out}")


if __name__ == "__main__":
    main()
