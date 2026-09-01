#!/usr/bin/env python3
"""Fail unless every floating-point tensor in a Lightning checkpoint is finite."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    checked = 0
    for name, value in state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            checked += 1
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite tensor: {name}")
    if checked == 0:
        raise RuntimeError("checkpoint contains no floating-point state tensors")
    print(f"[finite] {args.checkpoint}: {checked} tensors OK")


if __name__ == "__main__":
    main()
