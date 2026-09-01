from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAFE = (ROOT / "experiments/run_chime3_baseline_suite_lr05_safe.sh").read_text()


def test_representation_stages_use_finite_not_asr_convergence_gate():
    assert "*fpkd-stablekl-dfkd*|*fpkd-stablekl-frkd*|*carl-feature*" in SAFE
    assert 'gate_checkpoint "$name" 1.01' in SAFE
    assert 'gate_checkpoint "$name" 0.95' in SAFE


def test_safe_queue_runs_all_methods_and_final_evaluation():
    for token in (
        "paper-chime3-vanilla",
        "paper-chime3-kdbe",
        "paper-chime3-symmetric",
        "paper-chime3-guided",
        "paper-chime3-sctc",
        "paper-chime3-fpkd",
        "paper-chime3-carl",
        "paper-chime3-crctc",
        "evaluate_chime3_baseline_suite_lr05.sh",
    ):
        assert token in SAFE
