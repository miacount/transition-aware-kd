#!/usr/bin/env python3
"""Mix LibriSpeech clean speech with DEMAND noise at controlled SNRs.

The same utterance, DEMAND environment, channel, and noise crop are reused at
every SNR.  This yields a paired additive-noise stress test suitable for the
teacher/student mechanism analysis; it is not a standard LibriSpeech split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_chime3_snr_stress import (
    active_speech_mask,
    mix_at_snr,
    normalize_active_rms,
    read_mono,
    snr_tag,
    write_manifest,
)


ENVIRONMENTS = ("TBUS", "PCAFETER", "SPSQUARE", "STRAFFIC")
DEMAND_DOI = "10.5281/zenodo.1227121"


def load_manifest(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return sorted(rows, key=lambda row: Path(row["audio_filepath"]).stem)


def find_environment_noise(noise_root: Path, environment: str, channel: int) -> Path:
    channel_names = {
        f"ch{channel:02d}.wav",
        f"ch{channel}.wav",
        f"{environment}_ch{channel:02d}.wav",
        f"{environment}_ch{channel}.wav",
    }
    candidates = [
        path
        for path in noise_root.rglob("*.wav")
        if environment.lower() in str(path.parent).lower()
        and path.name.lower() in {name.lower() for name in channel_names}
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {environment} channel-{channel} WAV under {noise_root}, "
            f"found {len(candidates)}: {candidates[:10]}"
        )
    return candidates[0].resolve()


def deterministic_crop_start(
    utterance_id: str, environment: str, seed: int, available: int, needed: int
) -> int:
    if needed > available:
        raise ValueError(
            f"noise recording shorter than utterance: {available} < {needed} samples"
        )
    span = available - needed
    if span == 0:
        return 0
    digest = hashlib.sha256(
        f"{seed}|{utterance_id}|{environment}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (span + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/dev_clean.json"))
    parser.add_argument(
        "--noise-root", type=Path, default=Path("data/noise/DEMAND/extracted")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/librispeech_dev_clean_demand")
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=(10.0, 0.0, -5.0, -10.0))
    parser.add_argument("--channel", type=int, default=1, choices=range(1, 17))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--active-threshold-db", type=float, default=35.0)
    parser.add_argument("--speech-active-dbfs", type=float, default=-40.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    noise_root = args.noise_root.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_manifest(manifest_path, args.limit)
    noise_paths = {
        environment: find_environment_noise(noise_root, environment, args.channel)
        for environment in ENVIRONMENTS
    }
    noise_audio = {
        environment: read_mono(path, args.sample_rate)
        for environment, path in noise_paths.items()
    }

    missing_audio = [row["audio_filepath"] for row in rows if not Path(row["audio_filepath"]).is_file()]
    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} manifest audio files are missing; first: {missing_audio[0]}"
        )
    environments = Counter(ENVIRONMENTS[index % 4] for index in range(len(rows)))
    print(
        f"preflight OK: utterances={len(rows)}, environments={dict(environments)}, "
        f"SNRs={list(args.snrs)}, channel={args.channel}"
    )
    for environment, path in noise_paths.items():
        print(
            f"  {environment}: {path} "
            f"({len(noise_audio[environment]) / args.sample_rate:.1f} s)"
        )
    if args.preflight_only:
        return

    conditions = ["clean"] + [snr_tag(value) for value in args.snrs]
    if not args.overwrite:
        collisions = [
            output_dir / condition
            for condition in conditions
            if (output_dir / condition).exists()
            and any((output_dir / condition).iterdir())
        ]
        if collisions:
            raise FileExistsError(
                "output directories are not empty; use --overwrite: "
                + ", ".join(str(path) for path in collisions)
            )

    manifests: dict[str, list[dict]] = {condition: [] for condition in conditions}
    for index, source_row in enumerate(rows):
        source_path = Path(source_row["audio_filepath"]).resolve()
        utterance_id = source_path.stem
        environment = ENVIRONMENTS[index % len(ENVIRONMENTS)]
        speech = read_mono(source_path, args.sample_rate)
        active_mask = active_speech_mask(
            speech,
            args.sample_rate,
            threshold_db=args.active_threshold_db,
        )
        speech, speech_gain = normalize_active_rms(
            speech, active_mask, args.speech_active_dbfs
        )
        noise_recording = noise_audio[environment]
        crop_start = deterministic_crop_start(
            utterance_id,
            environment,
            args.seed,
            len(noise_recording),
            len(speech),
        )
        noise = noise_recording[crop_start : crop_start + len(speech)].copy()

        common = {
            "duration": round(len(speech) / args.sample_rate, 6),
            "text": source_row["text"],
            "utterance_id": utterance_id,
            "environment": environment,
            "source_clean": str(source_path),
            "source_noise": str(noise_paths[environment]),
            "noise_crop_start_sample": crop_start,
            "noise_crop_start_seconds": crop_start / args.sample_rate,
            "noise_channel": args.channel,
            "mix_seed": args.seed,
            "speech_gain": speech_gain,
            "active_speech_definition": (
                f"25ms/10ms frame RMS within {args.active_threshold_db:g} dB "
                "of 95th-percentile frame RMS"
            ),
        }
        clean_output_path = output_dir / "clean" / environment / f"{utterance_id}.flac"
        clean_output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            clean_output_path,
            speech.astype(np.float32),
            args.sample_rate,
            format="FLAC",
            subtype="PCM_16",
        )
        manifests["clean"].append(
            {
                **common,
                "audio_filepath": str(clean_output_path.resolve()),
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
                    f"{utterance_id} {condition}: achieved "
                    f"{diagnostics['achieved_snr_db']:.4f} dB"
                )
            output_path = output_dir / condition / environment / f"{utterance_id}.flac"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                mixture.astype(np.float32),
                args.sample_rate,
                format="FLAC",
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

        completed = index + 1
        if completed % 100 == 0 or completed == len(rows):
            print(f"generated {completed}/{len(rows)} utterances", flush=True)

    manifest_dir = output_dir / "manifests"
    for condition, condition_rows in manifests.items():
        write_manifest(manifest_dir / f"{condition}.json", condition_rows)
    metadata = {
        "description": "Paired LibriSpeech dev-clean + DEMAND additive-noise stress test",
        "diagnostic_not_standard_librispeech": True,
        "source_manifest": str(manifest_path),
        "demand_doi": DEMAND_DOI,
        "utterances": len(rows),
        "hours_per_condition": sum(float(row["duration"]) for row in rows) / 3600.0,
        "environments": dict(environments),
        "environment_order": list(ENVIRONMENTS),
        "snrs_db": list(args.snrs),
        "noise_channel": args.channel,
        "sample_rate": args.sample_rate,
        "active_threshold_db": args.active_threshold_db,
        "speech_active_dbfs": args.speech_active_dbfs,
        "mix_seed": args.seed,
        "assignment": "sorted utterance ID, four-environment round-robin",
        "crop": "SHA-256(seed, utterance ID, environment) mapped to valid crop starts",
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote paired stress-test data under {output_dir}")


if __name__ == "__main__":
    main()
