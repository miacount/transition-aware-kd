import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_chime3_snr_stress.py"
SPEC = importlib.util.spec_from_file_location("prepare_chime3_snr_stress", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mix_at_snr_hits_requested_active_region_snr():
    rng = np.random.default_rng(7)
    speech = rng.normal(size=32000)
    noise = rng.normal(size=32000)
    active = np.zeros(32000, dtype=bool)
    active[4000:28000] = True
    for target in (5.0, -5.0, -10.0):
        mixture, diagnostics = MODULE.mix_at_snr(speech, noise, active, target)
        assert mixture.shape == speech.shape
        assert abs(diagnostics["achieved_snr_db"] - target) < 1e-10
        assert np.max(np.abs(mixture)) <= 0.9990001


def test_balanced_items_is_deterministic_and_balanced():
    items = []
    for utterance in range(10):
        for environment in MODULE.ENVIRONMENTS:
            items.append(
                {
                    "speaker": "S",
                    "wsj_name": f"U{utterance:02d}",
                    "environment": environment,
                }
            )
    selected = MODULE.balanced_items(reversed(items))
    environments = [item["environment"] for item in selected]
    assert environments == [
        "BUS",
        "CAF",
        "PED",
        "STR",
        "BUS",
        "CAF",
        "PED",
        "STR",
        "BUS",
        "CAF",
    ]


def test_snr_tag():
    assert MODULE.snr_tag(5) == "snr_p5"
    assert MODULE.snr_tag(-5) == "snr_m5"
    assert MODULE.snr_tag(-7.5) == "snr_m7p5"
