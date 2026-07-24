#!/usr/bin/env python
"""Prepare TED-LIUM release 3 for the KD pipeline.

Reads the official release layout:
  <root>/data/{sph,stm}            -> train talks (~452h)
  <root>/legacy/dev/{sph,stm}      -> standard dev set
  <root>/legacy/test/{sph,stm}     -> standard test set

For each STM segment: normalize the transcript to the repo convention
(lowercase, clitic tokens like "it 's" joined to "it's"), cut the segment
from the .sph with sox into a 16k mono wav, and emit NeMo manifests
{audio_filepath, duration, text} (absolute paths, one JSON per line).

Segments dropped: scoring-ignored ("ignore_time_segment_in_scoring"),
containing <unk>, out of --min-dur/--max-dur, or with residual characters
outside [a-z' ]. Train is subsampled to --train-hours with a fixed seed
(uniform over segments, so speaker/talk diversity is preserved).

Usage:
  python scripts/prepare_tedlium.py --root data/TEDLIUM_release-3 \
      --out-dir data/tedlium --train-hours 100 --workers 16
"""
import argparse
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")


def normalize_stm_text(tokens):
    """lowercase + join clitic tokens ("it 's" -> "it's"). Returns None if the
    segment should be dropped (contains <unk> or chars outside the charset)."""
    words = []
    for tok in tokens:
        tok = tok.lower()
        if tok == "<unk>":
            return None
        if tok.startswith("'") and words:
            words[-1] += tok
        else:
            words.append(tok)
    text = " ".join(words)
    if not text or set(text) - ALLOWED:
        return None
    return text


def parse_stm_dir(stm_dir, sph_dir):
    segs, n_ignored, n_dropped = [], 0, 0
    for stm in sorted(Path(stm_dir).glob("*.stm")):
        sph = Path(sph_dir) / (stm.stem + ".sph")
        if not sph.exists():
            print(f"  [warn] missing sph for {stm.name}, skipping talk")
            continue
        for line in stm.open():
            parts = line.split()
            if len(parts) < 7:
                continue
            talk, _chan, _spk, t0, t1, _label = parts[:6]
            tokens = parts[6:]
            if "ignore_time_segment_in_scoring" in tokens:
                n_ignored += 1
                continue
            text = normalize_stm_text(tokens)
            if text is None:
                n_dropped += 1
                continue
            t0, t1 = float(t0), float(t1)
            segs.append({"sph": str(sph), "talk": talk, "t0": t0, "t1": t1,
                         "text": text, "duration": round(t1 - t0, 3)})
    return segs, n_ignored, n_dropped


def cut_one(job):
    seg, wav_path = job
    if not Path(wav_path).exists():
        cmd = ["sox", seg["sph"], "-r", "16000", "-c", "1", wav_path,
               "trim", f"{seg['t0']:.3f}", f"={seg['t1']:.3f}"]
        subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


def process_split(name, segs, out_dir, manifest_path, workers):
    wav_dir = Path(out_dir) / name
    wav_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, seg in enumerate(segs):
        wav = wav_dir / f"{seg['talk']}_{i:05d}.wav"
        jobs.append((seg, str(wav)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, _ in enumerate(ex.map(cut_one, jobs, chunksize=64)):
            if (n + 1) % 5000 == 0:
                print(f"  [{name}] cut {n + 1}/{len(jobs)}")
    with open(manifest_path, "w") as out:
        for seg, wav in jobs:
            row = {"audio_filepath": str(Path(wav).resolve()),
                   "duration": seg["duration"], "text": seg["text"]}
            out.write(json.dumps(row) + "\n")
    hours = sum(s["duration"] for s in segs) / 3600
    print(f"  [{name}] {len(segs)} segments, {hours:.1f}h -> {manifest_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TEDLIUM_release-3")
    ap.add_argument("--out-dir", default="data/tedlium")
    ap.add_argument("--train-hours", type=float, default=100.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)

    splits = {
        "train": (root / "data" / "stm", root / "data" / "sph"),
        "dev": (root / "legacy" / "dev" / "stm", root / "legacy" / "dev" / "sph"),
        "test": (root / "legacy" / "test" / "stm", root / "legacy" / "test" / "sph"),
    }
    for name, (stm_dir, sph_dir) in splits.items():
        if not stm_dir.exists():
            raise FileNotFoundError(f"{name}: {stm_dir} not found — check --root")

    for name, (stm_dir, sph_dir) in splits.items():
        segs, n_ign, n_drop = parse_stm_dir(stm_dir, sph_dir)
        n0 = len(segs)
        segs = [s for s in segs if args.min_dur <= s["duration"] <= args.max_dur]
        print(f"[{name}] {n0} parsed (+{n_ign} scoring-ignored, +{n_drop} unk/charset-dropped), "
              f"{len(segs)} after {args.min_dur}-{args.max_dur}s filter, "
              f"{sum(s['duration'] for s in segs) / 3600:.1f}h")
        if name == "train":
            rng = random.Random(args.seed)
            rng.shuffle(segs)
            picked, total = [], 0.0
            for s in segs:
                if total >= args.train_hours * 3600:
                    break
                picked.append(s)
                total += s["duration"]
            segs = sorted(picked, key=lambda s: (s["talk"], s["t0"]))
            print(f"[train] subsampled to {len(segs)} segments, {total / 3600:.1f}h (seed {args.seed})")
            manifest = f"data/tedlium_train_{int(args.train_hours)}h.json"
        else:
            manifest = f"data/tedlium_{name}.json"
        process_split(name, segs, args.out_dir, manifest, args.workers)


if __name__ == "__main__":
    main()
