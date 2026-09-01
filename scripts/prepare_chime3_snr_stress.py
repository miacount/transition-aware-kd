#!/usr/bin/env python3
"""Create a controlled CHiME-3 background-noise SNR stress-test set.

The script uses clean BTH utterances and the *official* CHiME-3 background
recordings referenced by the simulated-set annotation.  Every source utterance
is assigned one of BUS/CAF/PED/STR in a deterministic balanced round-robin.
The annotation-defined noise recording and crop are then reused at every SNR,
so comparisons across noise levels are paired.

This is a controlled diagnostic set, not an official CHiME-3 evaluation split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


ENVIRONMENTS = ("BUS", "CAF", "PED", "STR")


def normalize_text(text: str) -> str:
    text = text.lower().replace("\\'", "'").replace("\\", " ")
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"(^|\s)'+|'+($|\s)", " ", text)
    return " ".join(text.split())


def snr_tag(snr_db: float) -> str:
    sign = "p" if snr_db >= 0 else "m"
    value = abs(float(snr_db))
    number = f"{value:g}".replace(".", "p")
    return f"snr_{sign}{number}"


def read_mono(path: Path, expected_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise ValueError(f"{path}: expected {expected_sr} Hz, found {sr} Hz")
    if audio.ndim != 1:
        raise ValueError(f"{path}: expected mono WAV, found shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: contains non-finite samples")
    return audio.astype(np.float64, copy=False)


def active_speech_mask(
    speech: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    threshold_db: float = 35.0,
) -> np.ndarray:
    """Return a reproducible energy-based speech-active sample mask.

    A frame is active when its RMS lies within ``threshold_db`` of the 95th
    percentile frame RMS.  The definition is deliberately simple and is saved
    in the output metadata so the reported SNR can be reproduced exactly.
    """
    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(speech) < frame:
        power = np.mean(np.square(speech))
        if power <= 0:
            raise ValueError("silent clean utterance")
        return np.ones(len(speech), dtype=bool)

    starts = np.arange(0, len(speech) - frame + 1, hop)
    rms = np.sqrt(
        np.array([np.mean(np.square(speech[s : s + frame])) for s in starts])
        + 1e-20
    )
    reference = float(np.percentile(rms, 95.0))
    if reference <= 1e-9:
        raise ValueError("silent clean utterance")
    active_frames = rms >= reference * 10.0 ** (-threshold_db / 20.0)
    mask = np.zeros(len(speech), dtype=bool)
    for start, active in zip(starts, active_frames):
        if active:
            mask[start : start + frame] = True
    if mask.mean() < 0.01:
        raise ValueError("active-speech detector selected less than 1% of samples")
    return mask


def normalize_active_rms(
    speech: np.ndarray, mask: np.ndarray, target_dbfs: float
) -> tuple[np.ndarray, float]:
    rms = math.sqrt(float(np.mean(np.square(speech[mask]))))
    if rms <= 1e-12:
        raise ValueError("zero active-speech RMS")
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    return speech * gain, gain


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    active_mask: np.ndarray,
    snr_db: float,
    peak_limit: float = 0.999,
) -> tuple[np.ndarray, dict[str, float]]:
    if speech.shape != noise.shape:
        raise ValueError(f"speech/noise shape mismatch: {speech.shape} vs {noise.shape}")
    noise = noise - float(np.mean(noise))
    speech_power = float(np.mean(np.square(speech[active_mask])))
    noise_power = float(np.mean(np.square(noise[active_mask])))
    if speech_power <= 0 or noise_power <= 0:
        raise ValueError("speech or noise has zero active-region power")
    noise_gain = math.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    noise_component = noise_gain * noise
    mixture = speech + noise_component
    peak = float(np.max(np.abs(mixture)))
    output_gain = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    mixture *= output_gain
    speech_component = speech * output_gain
    noise_component *= output_gain
    achieved = 10.0 * math.log10(
        float(np.mean(np.square(speech_component[active_mask])))
        / float(np.mean(np.square(noise_component[active_mask])))
    )
    return mixture, {
        "noise_gain": noise_gain,
        "output_gain": output_gain,
        "prelimit_peak": peak,
        "achieved_snr_db": achieved,
    }


def balanced_items(items: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for item in items:
        key = (item["speaker"], item["wsj_name"])
        grouped[key][item["environment"].upper()] = item
    selected = []
    for index, key in enumerate(sorted(grouped)):
        environment = ENVIRONMENTS[index % len(ENVIRONMENTS)]
        if environment not in grouped[key]:
            raise ValueError(f"{key}: missing {environment} simulated annotation")
        selected.append(grouped[key][environment])
    return selected


def find_background(noise_dir: Path, stem: str, channel: int) -> Path:
    candidates = (
        noise_dir / f"{stem}.CH{channel}.wav",
        noise_dir / f"{stem}_CH{channel}.wav",
        noise_dir / f"{stem}.wav",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing official CHiME-3 background for {stem}, channel {channel}; "
        f"tried: {', '.join(str(path) for path in candidates)}"
    )


def crop_noise(
    background: np.ndarray,
    start_seconds: float,
    length: int,
    sample_rate: int,
) -> np.ndarray:
    start = round(start_seconds * sample_rate)
    end = start + length
    if start < 0 or end > len(background):
        raise ValueError(
            f"noise crop [{start}:{end}] exceeds background length {len(background)}"
        )
    return background[start:end].copy()


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=Path("data/chime/CHiME3"))
    parser.add_argument("--split", choices=("dt05", "et05"), default="dt05")
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help="official CHiME-3 16 kHz backgrounds directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/chime3_snr_stress")
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=(5.0, -5.0, -10.0))
    parser.add_argument("--channel", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--active-threshold-db", type=float, default=35.0)
    parser.add_argument("--speech-active-dbfs", type=float, default=-30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_root = args.corpus_root.resolve()
    noise_dir = (
        args.noise_dir.resolve()
        if args.noise_dir is not None
        else corpus_root / "data/audio/16kHz/backgrounds"
    )
    annotation_path = corpus_root / "data/annotations" / f"{args.split}_simu.json"
    isolated_dir = corpus_root / "data/audio/16kHz/isolated"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not noise_dir.is_dir():
        raise FileNotFoundError(
            f"official CHiME-3 backgrounds directory not found: {noise_dir}\n"
            "Supply it with --noise-dir. Residual noise estimated from noisy speech "
            "is intentionally not accepted for this paper-grade generator."
        )

    items = balanced_items(json.loads(annotation_path.read_text(encoding="utf-8")))
    if args.limit:
        items = items[: args.limit]

    missing = []
    resolved_backgrounds: dict[str, Path] = {}
    for item in items:
        stem = item["noise_wavfile"]
        if stem in resolved_backgrounds:
            continue
        try:
            resolved_backgrounds[stem] = find_background(noise_dir, stem, args.channel)
        except FileNotFoundError as error:
            missing.append(str(error))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} referenced background recordings are missing.\n{preview}"
        )

    environments = Counter(item["environment"].upper() for item in items)
    print(
        f"preflight OK: split={args.split}, utterances={len(items)}, "
        f"environments={dict(environments)}, backgrounds={len(resolved_backgrounds)}"
    )
    if args.preflight_only:
        return

    split_output = args.output_dir.resolve() / args.split
    conditions = ["clean"] + [snr_tag(value) for value in args.snrs]
    existing = [split_output / condition for condition in conditions]
    if not args.overwrite:
        collisions = [path for path in existing if path.exists() and any(path.iterdir())]
        if collisions:
            raise FileExistsError(
                "output directories are not empty; use --overwrite: "
                + ", ".join(str(path) for path in collisions)
            )

    manifests: dict[str, list[dict]] = {condition: [] for condition in conditions}
    background_cache: dict[Path, np.ndarray] = {}
    for index, item in enumerate(items, start=1):
        speaker = item["speaker"]
        wsj_name = item["wsj_name"]
        environment = item["environment"].upper()
        utterance_id = f"{speaker}_{wsj_name}_{environment}"
        clean_path = (
            isolated_dir
            / f"{args.split}_bth"
            / f"{speaker}_{wsj_name}_BTH.CH{args.channel}.wav"
        )
        speech_raw = read_mono(clean_path, args.sample_rate)
        active_mask = active_speech_mask(
            speech_raw,
            args.sample_rate,
            threshold_db=args.active_threshold_db,
        )
        speech, speech_gain = normalize_active_rms(
            speech_raw, active_mask, args.speech_active_dbfs
        )

        background_path = resolved_backgrounds[item["noise_wavfile"]]
        if background_path not in background_cache:
            background_cache[background_path] = read_mono(
                background_path, args.sample_rate
            )
        noise = crop_noise(
            background_cache[background_path],
            float(item["noise_start"]),
            len(speech),
            args.sample_rate,
        )
        annotated_duration = float(item["noise_end"]) - float(item["noise_start"])
        if abs(annotated_duration - len(speech) / args.sample_rate) > 0.03:
            raise ValueError(
                f"{utterance_id}: clean duration and annotated noise crop differ by "
                f"more than 30 ms ({len(speech) / args.sample_rate:.4f} vs "
                f"{annotated_duration:.4f} s)"
            )

        common = {
            "duration": round(len(speech) / args.sample_rate, 6),
            "text": normalize_text(item["dot"]),
            "utterance_id": utterance_id,
            "speaker": speaker,
            "wsj_name": wsj_name,
            "environment": environment,
            "source_clean": str(clean_path.resolve()),
            "source_noise": str(background_path.resolve()),
            "noise_start_seconds": float(item["noise_start"]),
            "channel": args.channel,
            "speech_gain": speech_gain,
            "active_speech_definition": (
                f"25ms/10ms frame RMS within {args.active_threshold_db:g} dB "
                "of 95th-percentile frame RMS"
            ),
        }

        clean_out = split_output / "clean" / f"{utterance_id}.wav"
        clean_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(clean_out, speech.astype(np.float32), args.sample_rate, subtype="PCM_16")
        manifests["clean"].append(
            {
                **common,
                "audio_filepath": str(clean_out.resolve()),
                "condition": "clean",
                "target_snr_db": None,
                "achieved_snr_db": None,
            }
        )

        for snr_db in args.snrs:
            condition = snr_tag(snr_db)
            mixture, diagnostics = mix_at_snr(
                speech, noise, active_mask, float(snr_db)
            )
            if abs(diagnostics["achieved_snr_db"] - snr_db) > 0.02:
                raise RuntimeError(
                    f"{utterance_id} {condition}: achieved SNR "
                    f"{diagnostics['achieved_snr_db']:.4f} dB"
                )
            output_path = split_output / condition / f"{utterance_id}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                mixture.astype(np.float32),
                args.sample_rate,
                subtype="PCM_16",
            )
            manifests[condition].append(
                {
                    **common,
                    "audio_filepath": str(output_path.resolve()),
                    "condition": condition,
                    "target_snr_db": float(snr_db),
                    **diagnostics,
                }
            )

        if index % 50 == 0 or index == len(items):
            print(f"generated {index}/{len(items)} utterances", flush=True)

    manifest_dir = split_output / "manifests"
    for condition, rows in manifests.items():
        write_manifest(manifest_dir / f"{condition}.json", rows)
    metadata = {
        "description": "Controlled CHiME-3 background-noise SNR stress test",
        "official_chime3_split": args.split,
        "diagnostic_not_official_eval": True,
        "utterances": len(items),
        "environments": dict(environments),
        "snrs_db": list(args.snrs),
        "channel": args.channel,
        "sample_rate": args.sample_rate,
        "active_threshold_db": args.active_threshold_db,
        "speech_active_dbfs": args.speech_active_dbfs,
        "selection": "sorted speaker/wsj_name, balanced BUS/CAF/PED/STR round-robin",
    }
    (manifest_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote audio and manifests under {split_output}")


if __name__ == "__main__":
    main()
