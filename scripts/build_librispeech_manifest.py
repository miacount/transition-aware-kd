#!/usr/bin/env python
"""Build NeMo-style JSON manifests from local LibriSpeech folders.

Expected layout:
  data/LibriSpeech/dev-clean/**/*.trans.txt
  data/LibriSpeech/dev-clean-processed/<utt_id>.wav

If the processed wav is missing, the script falls back to the original flac.
"""
import argparse
import json
from pathlib import Path


def read_duration(path):
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def build_manifest(split_dir, manifest_out, processed_dir=None, lowercase=True):
    split_dir = Path(split_dir)
    manifest_out = Path(manifest_out)
    processed_dir = Path(processed_dir) if processed_dir else None

    if not split_dir.exists():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    if processed_dir is not None and not processed_dir.exists():
        raise FileNotFoundError(f"processed directory not found: {processed_dir}")

    rows = []
    for trans_path in sorted(split_dir.glob("*/*/*.trans.txt")):
        for line in trans_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            utt_id, text = parts
            text = text.lower() if lowercase else text

            audio_path = processed_dir / f"{utt_id}.wav" if processed_dir else None
            if audio_path is None or not audio_path.exists():
                raw = trans_path.parent / f"{utt_id}.flac"
                if not raw.exists():
                    raise FileNotFoundError(f"missing audio for {utt_id}: {audio_path} or {raw}")
                audio_path = raw

            rows.append({
                "audio_filepath": str(audio_path.resolve()),
                "duration": round(read_duration(audio_path), 3),
                "text": text,
            })

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--processed_dir")
    ap.add_argument("--keep_case", action="store_true")
    args = ap.parse_args()

    rows = build_manifest(
        args.split_dir,
        args.manifest_out,
        processed_dir=args.processed_dir,
        lowercase=not args.keep_case,
    )
    total_hours = sum(r["duration"] for r in rows) / 3600.0
    print(f"utterances: {len(rows)}")
    print(f"hours     : {total_hours:.2f}")
    print(f"written   : {args.manifest_out}")


if __name__ == "__main__":
    main()
