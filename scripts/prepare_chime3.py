#!/usr/bin/env python3
"""Build NeMo JSONL manifests from the local CHiME-3 enhanced audio.

Only original-speed enhanced waveforms are included. The numbered annotation
shards and Picola speed perturbations are deliberately ignored.
"""

import argparse
import json
import re
import wave
from collections import Counter
from pathlib import Path


EXPECTED = {
    ("tr05", "real"): 1600,
    ("tr05", "simu"): 7138,
    ("dt05", "real"): 1640,
    ("dt05", "simu"): 1640,
    ("et05", "real"): 1320,
    ("et05", "simu"): 1320,
}
SPLIT_NAMES = {"tr05": "train", "dt05": "dev", "et05": "eval"}


def normalize_text(text: str) -> str:
    """Normalize CHiME's WSJ-style dot text for the alphabetic BPE."""
    text = text.lower().replace("\\'", "'")
    text = text.replace("\\", " ")
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"(^|\s)'+|'+($|\s)", " ", text)
    return " ".join(text.split())


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1:
            raise ValueError(f"unexpected audio format: {path}")
        return wav.getnframes() / wav.getframerate()


def build_rows(corpus_root: Path, split: str, condition: str):
    annotation = corpus_root / "data" / "annotations" / f"{split}_{condition}.json"
    enhanced = corpus_root / "data" / "audio" / "16kHz" / "enhanced"
    items = json.loads(annotation.read_text())
    expected = EXPECTED[(split, condition)]
    if len(items) != expected:
        raise ValueError(f"{annotation}: expected {expected} entries, found {len(items)}")

    rows = []
    seen = set()
    for item in items:
        environment = item["environment"].lower()
        utterance_id = f'{item["speaker"]}_{item["wsj_name"]}_{item["environment"]}'
        audio = enhanced / f"{split}_{environment}_{condition}" / f"{utterance_id}.wav"
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if utterance_id in seen:
            raise ValueError(f"duplicate utterance id in {annotation}: {utterance_id}")
        seen.add(utterance_id)
        text = normalize_text(item["dot"])
        if not text:
            raise ValueError(f"empty normalized transcript: {utterance_id}")
        rows.append({
            "audio_filepath": str(audio.resolve()),
            "duration": round(audio_duration(audio), 6),
            "text": text,
            "utterance_id": utterance_id,
            "speaker": item["speaker"],
            "wsj_name": item["wsj_name"],
            "environment": item["environment"],
            "condition": condition,
        })
    return sorted(rows, key=lambda row: row["utterance_id"])


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(row["duration"] for row in rows) / 3600
    envs = Counter(row["environment"] for row in rows)
    print(f"{path}: {len(rows)} utterances, {hours:.3f} h, environments={dict(envs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root", default="data/chime/CHiME3", type=Path,
        help="CHiME3 directory containing data/annotations and data/audio",
    )
    parser.add_argument("--output-dir", default="data", type=Path)
    args = parser.parse_args()

    for split, name in SPLIT_NAMES.items():
        real = build_rows(args.corpus_root, split, "real")
        simu = build_rows(args.corpus_root, split, "simu")
        write_jsonl(args.output_dir / f"chime3_{name}_real_enhanced.json", real)
        write_jsonl(args.output_dir / f"chime3_{name}_simu_enhanced.json", simu)
        write_jsonl(args.output_dir / f"chime3_{name}_enhanced.json", real + simu)


if __name__ == "__main__":
    main()
