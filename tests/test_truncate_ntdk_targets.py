import torch

from scripts.truncate_ntdk_targets import truncate_target


def test_truncate_moves_removed_selected_mass_into_tail():
    target = {
        "dark_ids": torch.tensor([[4, 2, 9, 7]], dtype=torch.int32),
        "dark_probs": torch.tensor([[0.4, 0.3, 0.2, 0.05]], dtype=torch.float16),
        "dark_tail_prob": torch.tensor([0.05], dtype=torch.float16),
        "dark_top_m": 4,
    }
    result = truncate_target(target, 2)
    assert result["dark_ids"].tolist() == [[4, 2]]
    torch.testing.assert_close(
        result["dark_probs"].float(), torch.tensor([[0.4, 0.3]]), atol=5e-4, rtol=0)
    torch.testing.assert_close(
        result["dark_tail_prob"].float(), torch.tensor([0.3]), atol=5e-4, rtol=0)
    torch.testing.assert_close(
        result["dark_probs"].float().sum(1) + result["dark_tail_prob"].float(),
        torch.ones(1), atol=5e-4, rtol=0)
    assert result["dark_top_m"] == 2
    assert result["source_dark_top_m"] == 4


def test_truncate_rejects_larger_top_m():
    target = {
        "dark_ids": torch.zeros(1, 3, dtype=torch.int32),
        "dark_probs": torch.zeros(1, 3, dtype=torch.float16),
        "dark_tail_prob": torch.ones(1, dtype=torch.float16),
    }
    try:
        truncate_target(target, 4)
    except ValueError as error:
        assert "[1, 3]" in str(error)
    else:
        raise AssertionError("expected ValueError")
