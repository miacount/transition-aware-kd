#!/usr/bin/env python3
"""Create deterministic NTDK ranking-control targets."""
import argparse
import json
from pathlib import Path
import torch


def read_manifest(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def permute_dark_probs(probs, seed):
    if probs.ndim != 2:
        raise ValueError(f"dark_probs must be rank 2, got {tuple(probs.shape)}")
    out = probs.clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for occurrence in range(probs.shape[0]):
        out[occurrence] = probs[occurrence, torch.randperm(probs.shape[1], generator=generator)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-in", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_in, manifest_out = Path(args.manifest_in), Path(args.manifest_out)
    out_dir = Path(args.out_dir)
    (manifest_out.parent / out_dir.name).mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_in)
    for index, row in enumerate(rows):
        source_rel = row.get("teacher_span_dark_path") or row.get("teacher_span_kd_path")
        if not source_rel:
            raise ValueError(f"row {index} has no span target path")
        source = Path(source_rel)
        if not source.is_absolute():
            source = manifest_in.parent / source
        rel = Path(out_dir.name) / f"{index:06d}.pt"
        output = manifest_out.parent / rel
        if args.resume and output.exists():
            cached = torch.load(output, map_location="cpu", weights_only=False)
            if cached.get("mode") == "ntdk_permuted_weights":
                row["teacher_span_dark_path"] = str(rel)
                continue
        target = torch.load(source, map_location="cpu", weights_only=False)
        required = {"dark_ids", "dark_probs", "dark_tail_prob", "mass3_probs"}
        missing = required - set(target)
        if missing:
            raise ValueError(f"{source} lacks {sorted(missing)}")
        original = target["dark_probs"]
        permuted = permute_dark_probs(original, args.seed + index)
        if not torch.equal(original.sort(1).values, permuted.sort(1).values):
            raise RuntimeError(f"probability multiset changed: {source}")
        target["dark_probs"] = permuted
        target["mode"] = "ntdk_permuted_weights"
        target["permutation_seed"] = int(args.seed)
        target["source_dark_path"] = str(source_rel)
        torch.save(target, output)
        row["teacher_span_dark_path"] = str(rel)
        if (index + 1) % 500 == 0:
            write_manifest(manifest_out, rows[:index + 1])
            print(f"[progress] {index + 1}/{len(rows)}", flush=True)
    write_manifest(manifest_out, rows)
    print(f"[done] {len(rows)} rows -> {manifest_out}")


if __name__ == "__main__":
    main()
