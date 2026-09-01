import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from permute_ntdk_targets import permute_dark_probs


def test_permuted_ntdk_preserves_each_probability_multiset():
    probs = torch.tensor([[.5, .3, .2], [.7, .2, .1]], dtype=torch.float16)
    actual = permute_dark_probs(probs, seed=17)
    assert torch.equal(actual.sort(1).values, probs.sort(1).values)
    assert not torch.equal(actual, probs)


def test_permuted_ntdk_is_deterministic():
    probs = torch.arange(64, dtype=torch.float32).view(2, 32)
    assert torch.equal(permute_dark_probs(probs, 20260818), permute_dark_probs(probs, 20260818))


def test_ablation_plan_has_exact_seven_new_runs():
    script = (ROOT / "experiments" / "run_ablation_7.sh").read_text()
    run_lines = [line for line in script.splitlines()
                 if line.startswith(("run_ablation ", "run_chime_ablation "))]
    assert len(run_lines) == 7
    assert "--kd-weight 0 --span-primary-mode mass3 --span-ntdk-weight 8" in script
    assert "permuted-ntdk" in script
