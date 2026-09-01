from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def test_full_config_is_standalone_and_chime_specific():
    cfg = OmegaConf.load(ROOT / "configs/student_base_chime3_full.yaml")
    assert cfg.get("defaults") is None
    assert cfg.model.tokenizer.dir == "tokenizer_1024"
    assert cfg.model.train_ds.manifest_filepath == "data/chime3_train_enhanced.json"
    assert cfg.model.validation_ds.manifest_filepath == "data/chime3_dev_enhanced.json"
    assert len(cfg.model.test_ds) == 4
    assert cfg.model.encoder.subsampling_factor == 4
    assert cfg.model.optim.sched.warmup_steps == 750
    for key in (
        "fpkd_temperature",
        "fpkd_bkl_weight",
        "fpkd_nbf_weight",
        "carl_teacher_dim",
        "carl_classifier_path",
        "carl_temperature",
        "carl_alpha",
        "carl_gamma",
        "carl_lambda",
    ):
        assert key in cfg.model
