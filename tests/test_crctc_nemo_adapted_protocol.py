from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CR = (ROOT / "experiments" / "run_crctc_nemo_adapted.sh").read_text()
CHAIN = (ROOT / "experiments" / "run_crctc_after_teacher.sh").read_text()


def test_crctc_uses_verified_nemo_mask_strength_not_collapsed_recipe():
    assert CR.count("--cr-ctc-time-factor 1.5") == 2
    assert "--cr-ctc-time-masks-scale 2.5" not in CR
    assert "--cr-ctc-time-width-scale 2.5" not in CR


def test_crctc_preserves_full_exposure_and_effective_batch():
    assert "nemo15-e100-b32a2-s1 100" in CR
    assert "--train-batch-size 32 --accumulate-grad-batches 2" in CR
    assert "nemo15-e50-b16a4-s1 50" in CR
    assert "--train-batch-size 16 --accumulate-grad-batches 4" in CR


def test_chain_requires_successful_teacher_export_before_crctc():
    assert "while pgrep" in CHAIN
    assert "[[ ! -s \"$TEACHER_OUT\" ]]" in CHAIN
    assert "run_crctc_nemo_adapted.sh all" in CHAIN
