#!/usr/bin/env python3
"""Derive smaller top-M+tail NTDK caches from an existing larger cache.

The source targets are ranked by teacher probability.  Classes removed from
the selected prefix are added to the existing residual tail, so no represented
probability mass is discarded and the teacher never needs to be run again.
"""
import argparse
import json
from pathlib import Path

import torch


def truncate_target(target, top_m):
    required = {"dark_ids", "dark_probs", "dark_tail_prob"}
    missing = required - set(target)
    if missing:
        raise ValueError(f"target lacks {sorted(missing)}")

    ids = target["dark_ids"]
    probs = target["dark_probs"]
    tail = target["dark_tail_prob"]
    if ids.ndim != 2 or probs.shape != ids.shape or tail.shape != ids.shape[:1]:
        raise ValueError(
            f"invalid dark target shapes: ids={tuple(ids.shape)}, "
            f"probs={tuple(probs.shape)}, tail={tuple(tail.shape)}")
    source_m = ids.shape[1]
    if not 0 < top_m <= source_m:
        raise ValueError(f"top_m must be in [1, {source_m}], got {top_m}")

    # Accumulate in float32 even though caches are normally float16.  Training
    # renormalizes the resulting M+1 buckets, but keeping their sum near one
    # avoids adding an unnecessary quantization-dependent difference.
    kept = probs[:, :top_m].float()
    new_tail = tail.float() + probs[:, top_m:].float().sum(dim=1)
    total = kept.sum(dim=1) + new_tail
    kept = kept / total.clamp_min(1e-30).unsqueeze(1)
    new_tail = new_tail / total.clamp_min(1e-30)

    result = dict(target)
    result["dark_ids"] = ids[:, :top_m].clone()
    result["dark_probs"] = kept.to(probs.dtype)
    result["dark_tail_prob"] = new_tail.to(tail.dtype)
    result["dark_top_m"] = int(top_m)
    result["source_dark_top_m"] = int(target.get("dark_top_m", source_m))
    result["mode"] = "raw_nontarget_topm_tail_truncated"
    return result


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def valid_cached(path, top_m):
    try:
        target = torch.load(path, map_location="cpu", weights_only=False)
        return (
            target.get("dark_top_m") == top_m
            and target.get("mode") == "raw_nontarget_topm_tail_truncated"
            and target["dark_ids"].shape[1] == top_m
            and target["dark_probs"].shape == target["dark_ids"].shape
        )
    except (OSError, KeyError, RuntimeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-in", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-m", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.manifest_in)
    output_dir = args.manifest_out.parent / args.out_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    masses = []

    for index, row in enumerate(rows):
        # Older/factorial manifests store support and dark targets together in
        # teacher_span_kd_path; newer d6 LS manifests split the two caches.
        # Both layouts carry the same dark_* tensors and are valid sources.
        source_rel = row.get("teacher_span_dark_path") or row.get("teacher_span_kd_path")
        if not source_rel:
            raise ValueError(
                f"row {index} has neither teacher_span_dark_path nor teacher_span_kd_path")
        source = Path(source_rel)
        if not source.is_absolute():
            source = args.manifest_in.parent / source
        rel = Path(args.out_dir.name) / f"{index:06d}.pt"
        output = args.manifest_out.parent / rel

        if args.resume and output.exists() and valid_cached(output, args.top_m):
            derived = torch.load(output, map_location="cpu", weights_only=False)
        else:
            source_target = torch.load(source, map_location="cpu", weights_only=False)
            derived = truncate_target(source_target, args.top_m)
            derived["source_dark_path"] = str(source_rel)
            torch.save(derived, output)

        selected_mass = derived["dark_probs"].float().sum(dim=1)
        masses.append(selected_mass)
        row["teacher_span_dark_path"] = str(rel)
        if (index + 1) % 500 == 0:
            print(f"[progress] M={args.top_m} {index + 1}/{len(rows)}", flush=True)

    write_rows(args.manifest_out, rows)
    mass = torch.cat(masses)
    print(
        f"[done] M={args.top_m} rows={len(rows)} occurrences={mass.numel()} "
        f"selected_mass_mean={mass.mean().item():.6f} "
        f"selected_mass_median={mass.median().item():.6f} "
        f"manifest={args.manifest_out}")


if __name__ == "__main__":
    main()
