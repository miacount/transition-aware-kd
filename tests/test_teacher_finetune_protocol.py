from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CFG = OmegaConf.load(ROOT / "configs" / "teacher_finetune_ted3.yaml")
SCRIPT = (ROOT / "scripts" / "finetune_teacher_ted3.py").read_text()


def test_teacher_finetune_uses_ted_train_dev_test_without_test_selection():
    assert CFG.train_ds.manifest_filepath == "data/tedlium3_train.json"
    assert CFG.validation_ds.manifest_filepath == "data/tedlium3_dev.json"
    assert CFG.test_ds.manifest_filepath == "data/tedlium3_test.json"
    assert CFG.exp_manager.checkpoint_callback_params.monitor == "val_wer"


def test_teacher_finetune_preserves_effective_batch_64():
    assert CFG.train_ds.batch_size * CFG.trainer.accumulate_grad_batches == 64


def test_teacher_finetune_is_low_lr_and_exports_best_checkpoint():
    assert CFG.optim.lr == 1.0e-4
    assert CFG.trainer.max_epochs == 10
    assert "callback.best_model_path" in SCRIPT
    assert "model.save_to" in SCRIPT
