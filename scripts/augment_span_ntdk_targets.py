#!/usr/bin/env python3
"""Add exact raw non-target dark targets while reusing stored Span-KD support.

No FB is recomputed. The frozen teacher is run in batches only for the missing
WHAT posterior, and compact top-M+tail files are referenced alongside the
existing delta=6 Span-KD targets.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_span_kd_targets import (  # noqa: E402
    load_audio,
    teacher_nontarget_topm,
    write_manifest,
)


def read_manifest(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def resolve(manifest, rel):
    path = Path(rel)
    return path if path.is_absolute() else Path(manifest).parent / path


def valid_dark(path, top_m, require_mass3=False):
    if not path.exists():
        return False
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        valid = (
            d.get("dark_top_m") == top_m
            and d["dark_ids"].shape[1] == top_m
            and d["dark_probs"].shape == d["dark_ids"].shape
            and d["dark_tail_prob"].shape[0] == d["dark_ids"].shape[0]
        )
        if require_mass3:
            valid = valid and d.get("mass3_probs") is not None and (
                d["mass3_probs"].shape == (d["dark_ids"].shape[0], 3))
        return valid
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-manifest", default=str(ROOT / "data/train_clean_100.teacher_only_d6.json"))
    ap.add_argument("--manifest-out", default=str(ROOT / "data/train_clean_100.teacher_only_d6_ntdk_m32.json"))
    ap.add_argument("--out-dir", default="span_ntdk_m32")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer_1024"))
    ap.add_argument("--top-m", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--with-mass3", action="store_true",
        help="also cache exact occupancy-pooled [blank, GT, non-target] mass")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.span_manifest)
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        row.pop("teacher_span_dark_path", None)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    manifest_dir = Path(args.manifest_out).parent
    out_dir = manifest_dir / Path(args.out_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve links to the reused Span target even when the augmented manifest
    # is written in a different directory (e.g. /tmp smoke tests).
    for row in rows:
        source_span = resolve(args.span_manifest, row["teacher_span_kd_path"]).resolve()
        row["teacher_span_kd_path"] = os.path.relpath(source_span, manifest_dir.resolve())

    reused = built = 0
    pending = []
    last_flushed = 0

    def flush_batch(items):
        nonlocal built
        if not items:
            return
        wavs = [torch.from_numpy(load_audio(row["audio_filepath"], sr)).float() for _, row, _ in items]
        lengths = torch.tensor([wav.numel() for wav in wavs], dtype=torch.long, device=args.device)
        padded = pad_sequence(wavs, batch_first=True).to(args.device)
        with torch.no_grad():
            log_probs, enc_len, _ = teacher(
                input_signal=padded, input_signal_length=lengths)
        for bi, (idx, row, out_path) in enumerate(items):
            span_path = resolve(args.manifest_out, row["teacher_span_kd_path"])
            span = torch.load(span_path, map_location="cpu", weights_only=False)
            n = int(span["num_tokens"])
            frames = int(enc_len[bi].item())
            support = span["support"][:n, :frames].float().numpy().T
            ids = [
                int(x) for x in tokenizer.text_to_ids(row["text"])
                if int(x) != blank
            ][:n]
            if len(ids) != n:
                raise ValueError(
                    f"token mismatch at row {idx}: target={n}, tokenizer={len(ids)}")
            if support.shape != (frames, n):
                raise ValueError(
                    f"support/frame mismatch at row {idx}: {support.shape} vs {(frames, n)}")
            lp = log_probs[bi, :frames].detach().cpu().float().numpy()
            result = teacher_nontarget_topm(
                lp, support, ids, blank, args.top_m,
                return_mass3=args.with_mass3)
            if args.with_mass3:
                dark_ids, dark_probs, dark_tail, mass3 = result
            else:
                dark_ids, dark_probs, dark_tail = result
            payload = {
                "dark_ids": torch.tensor(dark_ids, dtype=torch.int32),
                "dark_probs": torch.tensor(dark_probs, dtype=torch.float16),
                "dark_tail_prob": torch.tensor(dark_tail, dtype=torch.float16),
                "dark_top_m": int(args.top_m),
                "num_tokens": int(n),
                "teacher_frames": int(frames),
                "source_span_path": row["teacher_span_kd_path"],
                "mode": "raw_nontarget_topm_tail",
            }
            if args.with_mass3:
                payload["mass3_probs"] = torch.tensor(mass3, dtype=torch.float16)
            torch.save(payload, out_path)
            row["teacher_span_dark_path"] = str(Path(out_dir.name) / out_path.name)
            built += 1

    for idx, row in enumerate(rows):
        if not row.get("teacher_span_kd_path"):
            raise ValueError(f"row {idx} lacks teacher_span_kd_path")
        out_path = out_dir / f"{idx:06d}.pt"
        if args.resume and valid_dark(out_path, args.top_m, args.with_mass3):
            row["teacher_span_dark_path"] = str(Path(out_dir.name) / out_path.name)
            reused += 1
        else:
            pending.append((idx, row, out_path))
        if len(pending) >= args.batch_size:
            flush_batch(pending)
            pending = []
        done = built + reused
        if done - last_flushed >= args.flush_every:
            write_manifest(args.manifest_out, rows)
            print(f"[prog] done={done}/{len(rows)} built={built} reused={reused}", flush=True)
            last_flushed = done

    flush_batch(pending)
    write_manifest(args.manifest_out, rows)
    print("\n========== NTDK augmentation summary ==========")
    print(f"rows       : {len(rows)}")
    print(f"built      : {built}")
    print(f"reused     : {reused}")
    print(f"top-M      : {args.top_m} + tail")
    print(f"mass3      : {args.with_mass3}")
    print(f"FB runs    : 0 (reused stored Span-KD support)")
    print(f"manifest   : {args.manifest_out}")


if __name__ == "__main__":
    main()
