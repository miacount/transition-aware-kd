from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_target_builders_accept_exported_nemo_teacher():
    for name in ("build_span_kd_targets.py", "build_sctc_targets.py"):
        text = (ROOT / "scripts" / name).read_text()
        assert 'source.endswith(".nemo")' in text
        assert "restore_from(source)" in text


def test_adapted_suite_is_isolated_and_skips_teacher_independent_reruns():
    target_text = (ROOT / "experiments" / "prepare_ted3_adapted_teacher_targets.sh").read_text()
    train_text = (ROOT / "experiments" / "run_paper_main_ted3_adapted_teacher.sh").read_text()
    assert "best-teacher-ted3-adapted-s1.nemo" in target_text
    assert "adapted_teacher" in target_text
    assert "paper-ted3adapt" in train_text
    assert "kd-mode cr_ctc" not in train_text
    assert "paper-ted3full-no-kd" not in train_text
