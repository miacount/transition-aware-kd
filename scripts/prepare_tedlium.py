#!/usr/bin/env python
"""Prepare TED-LIUM releases 2 or 3 for the KD pipeline.

Reads the official release layout:
  <root>/data/{sph,stm}            -> train talks (~452h)
  <root>/legacy/dev/{sph,stm}      -> standard dev set
  <root>/legacy/test/{sph,stm}     -> standard test set

For each STM segment: normalize the transcript, cut it from the .sph into a
16kHz mono wav, and emit a NeMo JSON-lines manifest.

The ``kaldi`` normalization follows the public Lhotse/Kaldi recipe: noise tags
and ``<unk>`` markers are removed rather than discarding the complete segment,
and clitics are joined. Numbers/symbols are additionally verbalized for this
repository's fixed alphabetic LibriSpeech BPE. The legacy ``clean`` mode keeps
the old filtering behaviour for reproducibility.

Main-table usage:
  python scripts/prepare_tedlium.py --root data/TEDLIUM_release-3 \
      --out-dir data/tedlium3_full --manifest-prefix data/tedlium3 \
      --text-normalization kaldi --train-hours 0 \
      --min-dur 0.1 --max-dur 30 --keep-all-eval --workers 16

TED-LIUM2 usage:
  python scripts/prepare_tedlium.py --root data/TEDLIUM_release2 \
      --out-dir data/tedlium2_full --manifest-prefix data/tedlium2 \
      --text-normalization kaldi --train-hours 0 \
      --min-dur 0.1 --max-dur 30 --keep-all-eval --workers 16
"""
import argparse
import json
import random
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")


def _verbalize_number(match):
    """Verbalize a digit run while preserving surrounding letter chunks."""
    from num2words import num2words

    value = int(match.group(1))
    suffix = (match.group(2) or "").lower()
    if suffix in {"st", "nd", "rd", "th"}:
        return " " + num2words(value, to="ordinal").replace("-", " ") + " "
    if suffix == "s":
        # Spoken decades in TED transcripts (e.g. 1980s -> nineteen eighties).
        if 1000 <= value <= 2090 and value % 10 == 0:
            century, decade = divmod(value, 100)
            head = num2words(century).replace("-", " ")
            tail = num2words(decade).replace("-", " ")
            if decade == 0:
                return f" {head} hundreds "
            if tail.endswith("y"):
                tail = tail[:-1] + "ies"
            else:
                tail += "s"
            return f" {head} {tail} "
        return " " + num2words(value).replace("-", " ") + "s "
    return " " + num2words(value).replace("-", " ").replace(",", "") + " "


def normalize_stm_text_kaldi(tokens):
    """Kaldi/Lhotse-style TED normalization adapted to the alphabetic BPE."""
    text = " ".join(tokens).lower().replace("{noise}", "[noise]")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = text.replace("<unk>", " ")
    text = re.sub(r"\((?:[0-9]+)\)", "", text)  # pronunciation variants
    text = text.replace("&", " and ").replace("+", " plus ")
    text = text.replace("=", " equals ").replace("%", " percent ")
    text = text.replace("@", " at ").replace("#", " number ")
    text = text.replace("$", " dollar ")
    text = re.sub(r"(\d+)(st|nd|rd|th|s)?", _verbalize_number, text)
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"(\w+)\s+'(\w+)", r"\1'\2", text)
    text = re.sub(r"'\s+(\w+)", r"'\1", text)
    text = " ".join(text.split()).strip(" '")
    return text or None


def normalize_stm_text(tokens, mode="clean"):
    """Normalize one STM transcript; return None only when unusable/empty."""
    if mode == "kaldi":
        return normalize_stm_text_kaldi(tokens)

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


def parse_stm_dir(stm_dir, sph_dir, text_normalization="clean"):
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
            text = normalize_stm_text(tokens, text_normalization)
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
    ap.add_argument("--manifest-prefix",
                    help="write PREFIX_{train,dev,test}.json; legacy names if omitted")
    ap.add_argument("--train-hours", type=float, default=100.0,
                    help="0 keeps the full duration-filtered train set")
    ap.add_argument("--text-normalization", choices=("clean", "kaldi"),
                    default="clean")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--keep-all-eval", action="store_true",
                    help="do not apply the train duration filter to dev/test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)

    # Release 2 stores all splits directly below the corpus root. Release 3
    # stores full training data under data/ and dev/test under legacy/.
    if (root / "train" / "stm").is_dir():
        splits = {
            name: (root / name / "stm", root / name / "sph")
            for name in ("train", "dev", "test")
        }
    elif any((root / "train").glob("*.stm")):
        # Hugging Face's sharded release-2 mirror stores .stm and .sph files
        # together in each split directory.
        splits = {
            name: (root / name, root / name) for name in ("train", "dev", "test")
        }
    else:
        splits = {
            "train": (root / "data" / "stm", root / "data" / "sph"),
            "dev": (root / "legacy" / "dev" / "stm", root / "legacy" / "dev" / "sph"),
            "test": (root / "legacy" / "test" / "stm", root / "legacy" / "test" / "sph"),
        }
    for name, (stm_dir, _sph_dir) in splits.items():
        if not stm_dir.exists():
            raise FileNotFoundError(f"{name}: {stm_dir} not found — check --root")

    for name, (stm_dir, sph_dir) in splits.items():
        segs, n_ign, n_drop = parse_stm_dir(
            stm_dir, sph_dir, args.text_normalization
        )
        n0 = len(segs)
        if name == "train" or not args.keep_all_eval:
            segs = [s for s in segs if args.min_dur <= s["duration"] <= args.max_dur]
        print(f"[{name}] {n0} parsed (+{n_ign} scoring-ignored, "
              f"+{n_drop} normalization-dropped), {len(segs)} selected, "
              f"{sum(s['duration'] for s in segs) / 3600:.1f}h")
        if name == "train" and args.train_hours > 0:
            rng = random.Random(args.seed)
            rng.shuffle(segs)
            picked, total = [], 0.0
            for seg in segs:
                if total >= args.train_hours * 3600:
                    break
                picked.append(seg)
                total += seg["duration"]
            segs = sorted(picked, key=lambda seg: (seg["talk"], seg["t0"]))
            print(f"[train] subsampled to {len(segs)} segments, "
                  f"{total / 3600:.1f}h (seed {args.seed})")

        if args.manifest_prefix:
            manifest = f"{args.manifest_prefix}_{name}.json"
        elif name == "train":
            manifest = f"data/tedlium_train_{int(args.train_hours)}h.json"
        else:
            manifest = f"data/tedlium_{name}.json"
        process_split(name, segs, args.out_dir, manifest, args.workers)


if __name__ == "__main__":
    main()
