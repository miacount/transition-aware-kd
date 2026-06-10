#!/usr/bin/env python
"""Evaluate a student checkpoint on one or more manifests."""
import argparse
import os
import sys

import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import TransitionKDModel  # noqa: E402


def load_model(config_path, ckpt_path, device):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if model_cfg.get("test_ds") is not None:
            del model_cfg.test_ds
    model = TransitionKDModel(cfg=model_cfg, trainer=None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), cfg


def make_split_cfgs(cfg, manifests):
    if manifests:
        splits = []
        for item in manifests:
            name, path = item.split("=", 1) if "=" in item else (os.path.splitext(os.path.basename(item))[0], item)
            ds = cfg.model.validation_ds.copy()
            with open_dict(ds):
                ds.name = name
                ds.manifest_filepath = path
                ds.shuffle = False
            splits.append(ds)
        return splits
    return list(cfg.model.test_ds) if cfg.model.get("test_ds") is not None else [cfg.model.validation_ds]


@torch.no_grad()
def evaluate_split(model, ds_cfg, device):
    dl = model._make_dataloader_from_cfg(ds_cfg, shuffle=False)
    total_loss = 0.0
    total_items = 0
    total_wer_scores = 0
    total_wer_words = 0
    model.wer.reset()
    for batch in dl:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        log_probs, enc_len, _ = model.forward(batch["wavs"], batch["wav_lens"])
        loss = model.loss(
            log_probs=log_probs,
            targets=batch["tokens"],
            input_lengths=enc_len,
            target_lengths=batch["token_lens"],
        )
        batch_size = batch["wavs"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        # NeMo WER.update() overwrites state (not accumulates), so compute+reset per batch.
        model.wer.update(
            predictions=log_probs,
            predictions_lengths=enc_len,
            targets=batch["tokens"],
            targets_lengths=batch["token_lens"],
        )
        _, wer_num, wer_denom = model.wer.compute()
        model.wer.reset()
        total_wer_scores += int(wer_num.item())
        total_wer_words += int(wer_denom.item())
    wer = total_wer_scores / max(total_wer_words, 1)
    return total_loss / max(total_items, 1), wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", action="append", help="name=path, repeatable")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.config, args.ckpt, device)
    print(f"checkpoint: {args.ckpt}")
    for ds in make_split_cfgs(cfg, args.manifest):
        name = ds.get("name", os.path.splitext(os.path.basename(ds.manifest_filepath))[0])
        loss, wer = evaluate_split(model, ds, device)
        print(f"{name:12s} loss={loss:.4f} wer={wer * 100:.2f}%")


if __name__ == "__main__":
    main()
