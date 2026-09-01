#!/usr/bin/env python
"""Fine-tune the LibriSpeech NeMo CTC teacher on a TED-LIUM release.

The Hydra config selects the release (TED3 by default, TED2 via
``--config-name teacher_finetune_ted2``). The best checkpoint is exported as a
portable .nemo model, and evaluated once on the configured held-out test split.
"""

from pathlib import Path

import hydra
import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def _require_file(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(path)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="teacher_finetune_ted3",
)
def main(cfg: DictConfig) -> None:
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    for ds_name in ("train_ds", "validation_ds", "test_ds"):
        _require_file(str(cfg[ds_name].manifest_filepath))

    pl.seed_everything(int(cfg.seed), workers=True)
    trainer = pl.Trainer(logger=False, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    source = str(cfg.source_model)
    if source.endswith(".nemo"):
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(source)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(source)

    # Keep the source teacher's tokenizer/decoder and update only data and
    # optimization. This preserves exact vocabulary compatibility with all KD
    # target builders and students in this repository.
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create(OmegaConf.to_container(cfg.optim, resolve=True))
    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)

    trainer.fit(model)

    callback = trainer.checkpoint_callback
    best_path = callback.best_model_path if callback is not None else ""
    if not best_path:
        raise RuntimeError("training finished without a best validation checkpoint")
    logging.info(f"Restoring minimum-dev-WER teacher checkpoint: {best_path}")
    state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)

    output = Path(str(cfg.output_nemo))
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(output))
    logging.info(f"Exported adapted teacher: {output}")

    # Report the selected checkpoint on dev and test. Selection itself used
    # only val_wer during fit; test remains report-only.
    model.setup_validation_data(cfg.validation_ds)
    trainer.validate(model, dataloaders=model._validation_dl)
    model.setup_test_data(cfg.test_ds)
    trainer.test(model, dataloaders=model._test_dl)


if __name__ == "__main__":
    main()
