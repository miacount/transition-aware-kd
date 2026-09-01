from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chime_lr05_suite_restarts_both_students_with_same_schedule():
    suite = (ROOT / "experiments/run_chime3_students_lr05.sh").read_text()
    assert "paper-chime3-lr05-no-kd-s1 100" in suite
    assert "paper-chime3-lr05-mass3-w25-ntdk8-s1 100" in suite
    assert suite.count("--lr 0.5 --warmup-steps 750") == 2
    assert "--kd-weight 25" in suite
    assert "--span-primary-mode mass3" in suite
    assert "--span-ntdk-weight 8" in suite


def test_chime_lr05_evaluation_uses_new_experiment_names():
    evaluation = (ROOT / "experiments/evaluate_chime3_lr05.sh").read_text()
    assert "paper-chime3-lr05-no-kd-s1" in evaluation
    assert "paper-chime3-lr05-mass3-w25-ntdk8-s1" in evaluation
    assert "student_base_chime3_runtime.yaml" in evaluation
