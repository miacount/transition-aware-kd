"""Training entry point for clean CTC KD experiments."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from model import TransitionKDModel


@hydra.main(version_base=None, config_path="../configs", config_name="student_base")
def main(cfg):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    # Reproducibility: fix all RNGs (data shuffle, weight init, augmentation) so
    # ablation deltas reflect the config change, not seed noise. Set model.seed=N
    # to vary. Default 1. Single-run noise here is ~±0.2-0.3 WER, so effect sizes
    # below that need multiple seeds to be meaningful.
    seed = int(cfg.model.get("seed", 1))
    pl.seed_everything(seed, workers=True)
    trainer = pl.Trainer(logger=False, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    test_ds = cfg.model.get("test_ds", None)
    model_cfg = cfg.model.copy()
    if test_ds is not None:
        with open_dict(model_cfg):
            del model_cfg.test_ds

    model = TransitionKDModel(cfg=model_cfg, trainer=trainer)

    # Optional warm initialization from a pre-trained checkpoint (weights only, not optimizer state)
    init_ckpt = cfg.get("init_from_checkpoint", None)
    if init_ckpt:
        logging.info(f"Initializing model weights from: {init_ckpt}")
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            logging.warning(f"  Missing keys: {missing}")
        if unexpected:
            logging.warning(f"  Unexpected keys: {unexpected}")
        logging.info("  Weights loaded successfully (optimizer/epoch state discarded)")

    trainer.fit(model)

    if test_ds is not None:
        model.setup_test_data(test_ds)
        trainer.test(model, dataloaders=model._test_dl)


if __name__ == "__main__":
    main()
