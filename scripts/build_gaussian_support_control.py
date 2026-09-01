#!/usr/bin/env python3
"""Build a width-matched Gaussian temporal-support control.

The control keeps every delta=6 training quantity except the student-side
temporal support.  It replaces that support with a symmetric Gaussian
convolution of the delta=0 teacher occupancy.  The Gaussian sigma is selected
using training caches only so that its mean participation-ratio width matches
the delta=6 support.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(manifest, value):
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).parent / path


def normalized_support(support):
    support = support.float()
    return support / support.sum(dim=1, keepdim=True).clamp_min(1e-12)


def effective_width(support):
    support = normalized_support(support)
    return support.square().sum(dim=1).clamp_min(1e-12).reciprocal()


def gaussian_kernel(sigma, device=None):
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (offsets / max(float(sigma), 1e-6)).square())
    return kernel / kernel.sum()


def smooth_support(support, sigma):
    support = normalized_support(support)
    kernel = gaussian_kernel(sigma, support.device)
    radius = kernel.numel() // 2
    smoothed = F.conv1d(
        support.unsqueeze(1), kernel.view(1, 1, -1), padding=radius
    ).squeeze(1)
    return smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(1e-12)


def mean_width_from_payloads(paths):
    total = 0.0
    count = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        widths = effective_width(payload["support"])
        total += float(widths.sum())
        count += int(widths.numel())
    return total / max(count, 1), count


def calibration_supports(rows, manifest, count):
    if count >= len(rows):
        indices = np.arange(len(rows))
    else:
        indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    supports = []
    for index in indices:
        row = rows[int(index)]
        path = resolve(manifest, row["teacher_span_kd_path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        supports.append(payload["support"].float())
    return supports


def calibrated_sigma(supports, target_width, iterations=28):
    low, high = 0.01, 2.0
    for _ in range(iterations):
        sigma = 0.5 * (low + high)
        total = 0.0
        count = 0
        for support in supports:
            widths = effective_width(smooth_support(support, sigma))
            total += float(widths.sum())
            count += int(widths.numel())
        width = total / max(count, 1)
        if width < target_width:
            low = sigma
        else:
            high = sigma
    return 0.5 * (low + high)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta0-manifest", required=True)
    parser.add_argument("--delta6-manifest", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calibration-utts", type=int, default=4000)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows0 = read_jsonl(args.delta0_manifest)
    rows6 = read_jsonl(args.delta6_manifest)
    if len(rows0) != len(rows6):
        raise ValueError(f"manifest length mismatch: {len(rows0)} != {len(rows6)}")
    for index, (row0, row6) in enumerate(zip(rows0, rows6)):
        if row0["audio_filepath"] != row6["audio_filepath"] or row0["text"] != row6["text"]:
            raise ValueError(f"manifest alignment mismatch at row {index}")

    if args.limit is not None:
        rows0 = rows0[: args.limit]
        rows6 = rows6[: args.limit]

    d6_paths = [resolve(args.delta6_manifest, row["teacher_span_kd_path"]) for row in rows6]
    target_width, target_tokens = mean_width_from_payloads(d6_paths)
    if args.sigma is None:
        supports = calibration_supports(
            rows0, args.delta0_manifest, min(args.calibration_utts, len(rows0))
        )
        sigma = calibrated_sigma(supports, target_width)
    else:
        sigma = float(args.sigma)

    print(
        f"target delta6 width={target_width:.6f} over {target_tokens} tokens; "
        f"gaussian sigma={sigma:.6f}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = Path(args.manifest_out)
    output_rows = []
    width_sum = 0.0
    width_count = 0

    for index, (row0, row6) in enumerate(zip(rows0, rows6)):
        path0 = resolve(args.delta0_manifest, row0["teacher_span_kd_path"])
        path6 = resolve(args.delta6_manifest, row6["teacher_span_kd_path"])
        output = out_dir / f"{index:06d}.pt"

        if args.resume and output.exists():
            result = torch.load(output, map_location="cpu", weights_only=False)
        else:
            payload0 = torch.load(path0, map_location="cpu", weights_only=False)
            payload6 = torch.load(path6, map_location="cpu", weights_only=False)
            if payload0["support"].shape != payload6["support"].shape:
                raise ValueError(
                    f"support shape mismatch at row {index}: "
                    f"{tuple(payload0['support'].shape)} != {tuple(payload6['support'].shape)}"
                )
            result = dict(payload6)
            result["support"] = smooth_support(payload0["support"], sigma).to(torch.float16)
            result["mode"] = "gaussian_width_control"
            result["teacher_blank_penalty"] = 0.0
            result["gaussian_sigma"] = float(sigma)
            result["target_delta6_effective_width"] = float(target_width)
            result["source_delta0_support_path"] = str(row0["teacher_span_kd_path"])
            result["source_delta6_support_path"] = str(row6["teacher_span_kd_path"])
            temporary = output.with_suffix(".pt.tmp")
            torch.save(result, temporary)
            os.replace(temporary, output)

        widths = effective_width(result["support"])
        width_sum += float(widths.sum())
        width_count += int(widths.numel())
        output_row = dict(row6)
        output_row["teacher_span_kd_path"] = os.path.relpath(output, manifest_out.parent)
        output_rows.append(output_row)
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(rows0)}", flush=True)

    realized = width_sum / max(width_count, 1)
    write_jsonl(manifest_out, output_rows)
    print(
        f"realized gaussian width={realized:.6f}; delta={realized - target_width:+.6f}; "
        f"manifest={manifest_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
