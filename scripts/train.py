"""Training entry point for clean CTC KD experiments."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from model import TransitionKDModel


@hydra.main(version_base=None, config_path="../configs", config_name="student_base")
def main(cfg):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    trainer = pl.Trainer(logger=False, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    test_ds = cfg.model.get("test_ds", None)
    model_cfg = cfg.model.copy()
    if test_ds is not None:
        with open_dict(model_cfg):
            del model_cfg.test_ds

    model = TransitionKDModel(cfg=model_cfg, trainer=trainer)
    trainer.fit(model)

    if test_ds is not None:
        model.setup_test_data(test_ds)
        trainer.test(model, dataloaders=model._test_dl)


if __name__ == "__main__":
    main()
