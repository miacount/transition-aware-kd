from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_launcher_only_calls_safe_suite():
    text = (ROOT / "experiments/run_chime3_baselines.sh").read_text()
    assert "run_chime3_baseline_suite_lr05_safe.sh" in text
    assert "run_chime3_baseline_suite_lr05.sh" not in text.replace(
        "run_chime3_baseline_suite_lr05_safe.sh", ""
    )
