from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "experiments/resume_chime3_carl_crctc.sh").read_text()


def test_feature_stage_uses_finite_gate_instead_of_wer():
    feature_block = SCRIPT.split("# The full CARL stage", 1)[0]
    assert "check_checkpoint_finite.py" in feature_block
    assert "gate_asr" not in feature_block
    assert 'run_paper_train "$CARL_FEATURE" 10' in feature_block


def test_carl_full_gate_occurs_after_long_warmup():
    assert "--warmup-steps 1500" in SCRIPT
    assert "completed < 20" in SCRIPT
    assert 'gate_asr "$CARL_FULL" 0.95' in SCRIPT
    assert "--save-top-k -1" in SCRIPT
    assert "--last-n 10" in SCRIPT


def test_crctc_gate_and_final_evaluation_are_present():
    assert "--cr-ctc-time-factor 1.5" in SCRIPT
    assert "--cr-ctc-time-masks-scale 2.5" not in SCRIPT
    assert "completed < 12" in SCRIPT
    assert 'gate_asr "$CR_NAME" 0.95' in SCRIPT
    assert "evaluate_chime3_baseline_suite_lr05.sh" in SCRIPT
