from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_new_baseline_suite_declares_ctc_adaptations_and_temperatures():
    suite = (ROOT / "experiments/run_ted2_new_paper_baselines.sh").read_text()
    assert "frame-dkd-teacherargmax" in suite
    assert "--frame-dkd-temp 4" in suite
    assert "fpkd-stablekl" in suite
    assert "--fpkd-temp 1" in suite


def test_progressive_stages_scale_warmup_with_stage_length():
    suite = (ROOT / "experiments/run_ted2_new_paper_baselines.sh").read_text()
    assert 'FPKD_DFKD_WARMUP:-1000' in suite
    assert 'FPKD_FRKD_WARMUP:-1000' in suite
    assert 'FPKD_PKD_WARMUP:-8000' in suite
    assert "--warmup-steps 1000" in suite
    assert "--warmup-steps 5000" in suite


def test_carl_keeps_and_averages_last_ten_epochs():
    suite = (ROOT / "experiments/run_ted2_new_paper_baselines.sh").read_text()
    train = (ROOT / "experiments/train.sh").read_text()
    assert "--save-top-k -1" in suite
    assert "scripts/average_checkpoints.py" in suite
    assert "--last-n 10" in suite
    assert "exp_manager.checkpoint_callback_params.save_top_k" in train


def test_carl_cache_has_teacher_and_manifest_provenance():
    builder = (ROOT / "scripts/build_carl_targets.py").read_text()
