import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_librispeech_demand_snr.py"
SPEC = importlib.util.spec_from_file_location("prepare_librispeech_demand_snr", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_crop_start_is_deterministic_and_valid():
    first = MODULE.deterministic_crop_start("1272-128104-0000", "TBUS", 7, 1000, 100)
    second = MODULE.deterministic_crop_start("1272-128104-0000", "TBUS", 7, 1000, 100)
    assert first == second
    assert 0 <= first <= 900


def test_crop_start_changes_with_condition_identity():
    starts = {
        MODULE.deterministic_crop_start("utt", environment, 7, 100000, 100)
        for environment in MODULE.ENVIRONMENTS
    }
    assert len(starts) == len(MODULE.ENVIRONMENTS)


def test_environment_mapping_matches_paper_design():
    assert MODULE.ENVIRONMENTS == ("TBUS", "PCAFETER", "SPSQUARE", "STRAFFIC")
