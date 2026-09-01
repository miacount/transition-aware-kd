#!/usr/bin/env python3
"""Average the last N epoch checkpoints from one NeMo/Lightning experiment."""

import argparse
import re
from pathlib import Path

import torch


EPOCH_RE = re.compile(r"epoch=(\d+)(?:\D|$)")


def select_last_epoch_checkpoints(experiment_dir: Path, last_n: int):
    by_epoch = {}
    for path in experiment_dir.rglob("*.ckpt"):
        if path.name.endswith("last.ckpt"):
            continue
        match = EPOCH_RE.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        previous = by_epoch.get(epoch)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            by_epoch[epoch] = path
    selected = [by_epoch[e] for e in sorted(by_epoch)[-last_n:]]
    if len(selected) != last_n:
        raise RuntimeError(
            f"expected {last_n} distinct epoch checkpoints under {experiment_dir}, "
            f"found {len(selected)}; train with save_top_k=-1")
    return selected


def average_checkpoints(paths, output: Path):
    latest = torch.load(paths[-1], map_location="cpu", weights_only=False)
    reference = latest["state_dict"]
    sums = {
        key: torch.zeros_like(value, dtype=torch.float64)
        for key, value in reference.items()
        if torch.is_floating_point(value)
    }
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        if state.keys() != reference.keys():
            raise RuntimeError(f"state-dict key mismatch in {path}")
        for key, accumulator in sums.items():
            value = state[key]
            if value.shape != reference[key].shape:
                raise RuntimeError(f"shape mismatch for {key} in {path}")
            accumulator.add_(value.double())
    for key, accumulator in sums.items():
        reference[key] = (accumulator / len(paths)).to(reference[key].dtype)
    latest["checkpoint_average"] = {
        "count": len(paths),
        "sources": [str(path) for path in paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latest, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.last_n <= 0:
        parser.error("--last-n must be positive")
    paths = select_last_epoch_checkpoints(args.experiment_dir, args.last_n)
    average_checkpoints(paths, args.output)
    epochs = [EPOCH_RE.search(path.name).group(1) for path in paths]
    print("[average] epochs:", ", ".join(epochs))
    print(f"[average] output: {args.output}")


if __name__ == "__main__":
    main()
