import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from average_checkpoints import average_checkpoints, select_last_epoch_checkpoints


def test_select_and_average_last_epochs(tmp_path: Path):
    for epoch in range(12):
        torch.save(
            {"state_dict": {
                "weight": torch.tensor([float(epoch)]),
                "counter": torch.tensor(epoch, dtype=torch.long),
            }},
            tmp_path / f"model--val_wer=0.2-epoch={epoch}.ckpt",
        )
    paths = select_last_epoch_checkpoints(tmp_path, 10)
    epochs = [int(path.name.rsplit("epoch=", 1)[1].split(".")[0]) for path in paths]
    assert epochs == list(range(2, 12))
    output = tmp_path / "last10-avg.ckpt"
    average_checkpoints(paths, output)
    averaged = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.allclose(averaged["state_dict"]["weight"], torch.tensor([6.5]))
    # Non-floating state follows the latest checkpoint rather than being averaged.
    assert averaged["state_dict"]["counter"].item() == 11
    assert averaged["checkpoint_average"]["count"] == 10
