from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ours_only_bypasses_legacy_post_eval_and_keeps_protocol():
    text = (ROOT / "experiments/run_chime3_ours_only_lr05.sh").read_text()
    assert "experiments/train.sh" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--config-name student_base_chime3_runtime" in text
    assert "model.kd_mode=span_kd" in text
    assert "model.kd_weight=25" in text
    assert "model.span_primary_mode=mass3" in text
    assert "model.span_ntdk_weight=8" in text
    assert "model.optim.lr=0.5" in text
    assert "model.optim.sched.warmup_steps=750" in text
    assert "trainer.max_epochs=100" in text


def test_fixed_evaluators_use_materialized_config():
    for filename in (
        "experiments/run_chime3_ours_only_lr05.sh",
        "experiments/evaluate_chime3_ours_lr05_fixed.sh",
    ):
        text = (ROOT / filename).read_text()
        assert "configs/student_base_ted3.yaml" in text
        assert "configs/student_base_chime3_runtime.yaml" not in text
