from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]


def test_chime_runtime_declares_every_unconditional_train_override():
    required = {
        "frame_dkd_alpha",
        "frame_dkd_beta",
        "frame_dkd_temperature",
        "frame_dkd_warmup_epochs",
        "fpkd_temperature",
        "fpkd_bkl_weight",
        "fpkd_nbf_weight",
        "carl_teacher_dim",
        "carl_classifier_path",
        "carl_temperature",
        "carl_alpha",
        "carl_gamma",
        "carl_lambda",
    }
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="student_base_chime3_runtime")
    assert required <= set(cfg.model.keys())
    assert cfg.model.train_ds.manifest_filepath == "data/chime3_train_enhanced.json"
    assert cfg.model.validation_ds.manifest_filepath == "data/chime3_dev_enhanced.json"
    assert cfg.model.optim.sched.warmup_steps == 750
