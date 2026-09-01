from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "experiments/resume_chime3_baselines_after_guided.sh").read_text()


def test_guided_gate_runs_after_noam_warmup():
    assert "GATE_EPOCHS=12" in SCRIPT
    assert "GATE_MAX_WER=0.90" in SCRIPT
    assert "--warmup-steps 750" in SCRIPT
    assert 'run_paper_train "$NAME" "$GATE_EPOCHS"' in SCRIPT


def test_resume_preserves_exact_guided_recipe_and_continues_safe_suite():
    assert "--kd-mode guided" in SCRIPT
    assert "--temperature 1 --kd-weight 1" in SCRIPT
    assert "--lr 0.5" in SCRIPT
    assert "run_chime3_baseline_suite_lr05_safe.sh" in SCRIPT
    assert "latest_last_ckpt" in SCRIPT
