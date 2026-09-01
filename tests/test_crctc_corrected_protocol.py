from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "experiments" / "run_crctc_corrected.sh").read_text()


def test_lbs_crctc_keeps_effective_batch_and_full_exposure():
    assert "corrected-e100-b32a2-mask2p5-s1 100" in SCRIPT
    assert "--train-batch-size 32 --accumulate-grad-batches 2" in SCRIPT


def test_ted3_crctc_keeps_effective_batch_and_full_exposure():
    assert "corrected-e50-b16a4-mask2p5-s1 50" in SCRIPT
    assert "--train-batch-size 16 --accumulate-grad-batches 4" in SCRIPT


def test_official_crctc_loss_and_mask_settings_are_explicit():
    assert SCRIPT.count("--cr-ctc-weight 0.2 --cr-ctc-warm-step 2000") == 2
    assert SCRIPT.count("--cr-ctc-time-masks-scale 2.5 --cr-ctc-time-width-scale 2.5") == 2
