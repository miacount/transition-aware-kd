from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = (ROOT / "experiments/run_chime3_baseline_suite_lr05.sh").read_text()


def test_all_requested_methods_are_present():
    for mode in (
        "--kd-mode logit",
        "--blank-mode elimination",
        "--blank-mode symmetric",
        "--kd-mode guided",
        "--kd-mode sctc",
        "--kd-mode fpkd_dfkd",
        "--kd-mode fpkd_frkd",
        "--kd-mode fpkd_pkd",
        "--kd-mode carl_feature",
        "--kd-mode carl",
        "--kd-mode cr_ctc",
    ):
        assert mode in SUITE


def test_chime_safe_optimizer_settings_and_smoke_gate():
    assert "COMMON=(--config \"$CONFIG\" --lr 0.5 --warmup-steps 750)" in SUITE
    assert "run_guarded" in SUITE
    assert "gate_checkpoint" in SUITE
    assert "required <$max_wer" in SUITE


def test_known_failure_modes_are_not_reintroduced():
    assert "--cr-ctc-time-factor 1.5" in SUITE
    assert "--cr-ctc-time-masks-scale 2.5" not in SUITE
    assert "--cr-ctc-time-width-scale 2.5" not in SUITE
    assert "ctcft-lr005-wu100" in SUITE
    assert "--save-top-k -1" in SUITE
    assert "average_checkpoints.py" in SUITE


def test_target_preparation_is_resumable_and_count_checked():
    text = (ROOT / "experiments/prepare_chime3_baseline_targets.sh").read_text()
    assert text.count("--resume") == 2
    assert "expected 8738" in text
    assert "teacher_classifier.pt" in text
