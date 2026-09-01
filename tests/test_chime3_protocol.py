import json
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def test_chime_configs_keep_fair_student_comparison():
    cfg = OmegaConf.load(ROOT / "configs/student_base_chime3.yaml")
    assert cfg.model.optim.sched.warmup_steps == 750
    assert cfg.model.train_ds.batch_size == 32
    suite = (ROOT / "experiments/run_chime3_students.sh").read_text()
    assert "paper-chime3-no-kd-s1 100" in suite
    assert "paper-chime3-mass3-w25-ntdk8-s1 100" in suite
    assert "--kd-weight 25" in suite
    assert "--span-primary-mode mass3" in suite
    assert "--span-ntdk-weight 8" in suite


def test_chime_teacher_uses_train_dev_eval_without_eval_selection():
    cfg = OmegaConf.load(ROOT / "configs/teacher_finetune_chime3.yaml")
    assert cfg.train_ds.manifest_filepath == "data/chime3_train_enhanced.json"
    assert cfg.validation_ds.manifest_filepath == "data/chime3_dev_enhanced.json"
    assert cfg.test_ds.manifest_filepath == "data/chime3_eval_enhanced.json"
    assert cfg.train_ds.batch_size * cfg.trainer.accumulate_grad_batches == 64
    assert cfg.trainer.max_epochs == 10


def test_generated_chime_manifests_have_expected_sizes_when_present():
    expected = {
        "chime3_train_enhanced.json": 8738,
        "chime3_dev_enhanced.json": 3280,
        "chime3_eval_enhanced.json": 2640,
    }
    for name, count in expected.items():
        path = ROOT / "data" / name
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == count
        assert all(Path(row["audio_filepath"]).is_file() for row in rows)
        assert all(row["text"] == row["text"].lower() for row in rows)
        assert all(set(row["text"]) <= set("abcdefghijklmnopqrstuvwxyz' ") for row in rows)
