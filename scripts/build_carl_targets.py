#!/usr/bin/env python3
"""Cache frozen teacher encoder features and classifier for CARL/FPKD.

The input manifest should already contain the T=1 dense teacher posterior so
all posterior- and feature-based losses share exactly the same frozen teacher.
Each feature file stores the final encoder representation before the CTC
classifier as fp16 (T, D_teacher). The classifier artifact stores the frozen
1x1 Conv used by full CARL's auxiliary branch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence


def read_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)


def file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-in", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out-dir", default="data/carl_features")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=100)
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    teacher_path = Path(args.teacher).resolve()
    manifest_in_path = Path(args.manifest_in).resolve()
    teacher_sha256 = file_sha256(teacher_path)
    manifest_sha256 = file_sha256(manifest_in_path)
    teacher = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.teacher)
    teacher = teacher.to(args.device).eval()
    teacher.freeze()
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    decoder = teacher.decoder.decoder_layers[0]
    if not isinstance(decoder, torch.nn.Conv1d) or decoder.kernel_size != (1,):
        raise TypeError(f"expected Conv1d(k=1) teacher classifier, got {decoder}")

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[:args.limit]
    missing_dense = [i for i, row in enumerate(rows) if not row.get("teacher_frame_kd_path")]
    if missing_dense:
        raise ValueError(f"input manifest lacks dense posterior paths (first row {missing_dense[0]})")

    manifest_dir = Path(args.manifest_out).parent
    out_root = manifest_dir / Path(args.out_dir).name
    out_root.mkdir(parents=True, exist_ok=True)
    classifier_path = out_root / "teacher_classifier.pt"
    torch.save({
        "weight": decoder.weight.detach().cpu(),
        "bias": decoder.bias.detach().cpu(),
        "teacher_dim": int(decoder.weight.shape[1]),
        "vocab_size": int(decoder.weight.shape[0]),
        "teacher": str(teacher_path),
        "teacher_sha256": teacher_sha256,
        "source_manifest": str(manifest_in_path),
        "source_manifest_sha256": manifest_sha256,
        "cache_format": 2,
    }, classifier_path)

    def load_audio(path):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        return torch.from_numpy(wav)

    pending = []
    reused = 0
    for idx, row in enumerate(rows):
        rel = Path(out_root.name) / f"{idx:06d}.pt"
        target = manifest_dir / rel
        if args.resume and target.exists():
            try:
                cached = torch.load(target, map_location="cpu", weights_only=False)
                valid = ("features" in cached and cached["features"].ndim == 2
                         and cached["features"].shape[1] == decoder.weight.shape[1]
                         and cached.get("teacher_sha256") == teacher_sha256
                         and cached.get("source_manifest_sha256") == manifest_sha256
                         and cached.get("audio_filepath") == row["audio_filepath"]
                         and cached.get("dense_posterior_path")
                             == row["teacher_frame_kd_path"])
                if valid:
                    row["teacher_carl_path"] = str(rel)
                    reused += 1
                    continue
            except Exception as exc:
                print(f"[warn] rebuilding {target}: {exc}", flush=True)
        pending.append(idx)

    generated = 0
    frame_total = 0
    with torch.no_grad():
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            sigs = [load_audio(rows[i]["audio_filepath"]) for i in indices]
            lens = torch.tensor([x.numel() for x in sigs], device=args.device, dtype=torch.long)
            padded = pad_sequence(sigs, batch_first=True).to(args.device)
            processed, processed_len = teacher.preprocessor(input_signal=padded, length=lens)
            encoded, encoded_len = teacher.encoder(audio_signal=processed, length=processed_len)
            for local, idx in enumerate(indices):
                frames = int(encoded_len[local].item())
                features = encoded[local, :, :frames].transpose(0, 1).detach().cpu().to(torch.float16)
                rel = Path(out_root.name) / f"{idx:06d}.pt"
                torch.save({
                    "features": features,
                    "frames": frames,
                    "teacher_dim": int(features.shape[1]),
                    "teacher": str(teacher_path),
                    "teacher_sha256": teacher_sha256,
                    "source_manifest": str(manifest_in_path),
                    "source_manifest_sha256": manifest_sha256,
                    "audio_filepath": rows[idx]["audio_filepath"],
                    "dense_posterior_path": rows[idx]["teacher_frame_kd_path"],
                    "cache_format": 2,
                }, manifest_dir / rel)
                rows[idx]["teacher_carl_path"] = str(rel)
                generated += 1
                frame_total += frames
            if generated and generated % args.flush_every < len(indices):
                write_manifest(args.manifest_out, rows)
                print(f"[progress] generated={generated}/{len(pending)} reused={reused}", flush=True)

    write_manifest(args.manifest_out, rows)
    print(f"[done] rows={len(rows)} reused={reused} generated={generated} frames={frame_total}")
    print(f"[classifier] {classifier_path}")
    print(f"[manifest] {args.manifest_out}")


if __name__ == "__main__":
    main()
