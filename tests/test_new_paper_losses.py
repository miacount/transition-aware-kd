import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from data import collate_fn
from model import TransitionKDModel


def bare_model(vocab=5):
    model = TransitionKDModel.__new__(TransitionKDModel)
    torch.nn.Module.__init__(model)
    model.blank_id = vocab - 1
    return model


def test_binary_blank_kl_is_zero_for_identical_distributions():
    model = bare_model()
    probs = torch.tensor([[[.1, .1, .1, .1, .6], [.5, .1, .1, .1, .2]]])
    lp = probs.log()
    lens = torch.tensor([2])
    loss = model._binary_blank_kl(lp, lens, probs, lens, temperature=1.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_binary_blank_kl_is_finite_for_exact_zero_one_teacher_targets():
    model = bare_model()
    # Reduced-precision CTC caches commonly contain exact 0/1 probabilities.
    teacher = torch.tensor([[[0., 0., 0., 0., 1.],
                             [1., 0., 0., 0., 0.]]])
    logits = torch.tensor([[[0., 0., 0., 0., 4.],
                            [4., 0., 0., 0., 0.]]], requires_grad=True)
    lens = torch.tensor([2])
    loss = model._binary_blank_kl(
        logits.log_softmax(-1), lens, teacher, lens, temperature=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_fpkd_eliminates_blank_frame_conditional_nonblank_kl():
    model = bare_model()
    model.fpkd_temperature = 1.0
    model.fpkd_bkl_weight = 1.0
    model.fpkd_nbf_weight = 1.0
    # Same blank mass and teacher top-1 blank at every frame, but deliberately
    # incompatible conditional distributions among nonblank tokens.
    teacher = torch.tensor([[[.7, .1, .1, .0, .1], [.0, .1, .1, .1, .7]]])
    student = torch.tensor([[[.1, .7, .1, .0, .1], [.1, .1, .0, .1, .7]]])
    # first frame is nonblank in both; second is blank and must contribute BKL only
    lens = torch.tensor([2])
    total, bkl, nbf = model._fpkd_fkl_loss(student.clamp_min(1e-8).log(), lens, teacher, lens)
    assert torch.isfinite(total)
    # Altering only the blank frame's conditional NB distribution cannot change FKL.
    student2 = student.clone()
    student2[0, 1, :4] = torch.tensor([.0, .0, .3, .0])
    total2, bkl2, nbf2 = model._fpkd_fkl_loss(student2.clamp_min(1e-8).log(), lens, teacher, lens)
    assert torch.allclose(bkl, bkl2, atol=1e-6)
    assert torch.allclose(nbf, nbf2, atol=1e-6)
    assert torch.allclose(total, total2, atol=1e-6)


def test_fpkd_fkl_is_finite_for_exact_zero_one_teacher_targets():
    model = bare_model()
    model.fpkd_temperature = 1.0
    model.fpkd_bkl_weight = 1.0
    model.fpkd_nbf_weight = 1.0
    teacher = torch.tensor([[[0., 0., 0., 0., 1.],
                             [1., 0., 0., 0., 0.]]])
    logits = torch.tensor([[[0., 0., 0., 0., 4.],
                            [4., 0., 0., 0., 0.]]], requires_grad=True)
    lens = torch.tensor([2])
    total, bkl, nbf = model._fpkd_fkl_loss(
        logits.log_softmax(-1), lens, teacher, lens)
    assert torch.isfinite(total)
    assert torch.isfinite(bkl)
    assert torch.isfinite(nbf)
    total.backward()
    assert torch.isfinite(logits.grad).all()


def test_frame_dkd_zero_for_identical_distributions():
    model = bare_model()
    model.frame_dkd_alpha = 1.0
    model.frame_dkd_beta = 8.0
    model.frame_dkd_temperature = 4.0
    probs = torch.tensor([[[.1, .2, .1, .1, .5], [.6, .1, .1, .1, .1]]])
    lens = torch.tensor([2])
    loss = model._frame_dkd_loss(probs.log(), lens, probs, lens)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=2e-5)


def test_temperature_recovery_preserves_tiny_teacher_tail():
    model = bare_model()
    probs = torch.tensor([[[.999, 1e-8, 1e-16, 1e-24, 1e-32]]], dtype=torch.float32)
    actual = model._teacher_probs_at_temperature(probs, 4.0)
    expected = probs.pow(0.25)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    # The old 1e-12 log floor made all three smallest entries identical.
    assert actual[0, 0, 2] > actual[0, 0, 3] > actual[0, 0, 4] > 0


def test_temperature_recovery_does_not_invent_mass_for_cached_zeros():
    model = bare_model()
    probs = torch.tensor([[[.9, .1, 0.0, 0.0, 0.0]]])
    actual = model._teacher_probs_at_temperature(probs, 4.0)
    assert torch.equal(actual[..., 2:], torch.zeros_like(actual[..., 2:]))
    assert torch.allclose(actual.sum(-1), torch.ones_like(actual.sum(-1)))


def test_collate_carl_features():
    batch = [
        {"wav": torch.ones(4), "tokens": torch.tensor([1]),
         "carl_features": torch.ones(2, 3), "carl_teacher_frames": 2},
        {"wav": torch.ones(3), "tokens": torch.tensor([2, 1]),
         "carl_features": torch.ones(1, 3) * 2, "carl_teacher_frames": 1},
    ]
    out = collate_fn(batch)
    assert out["carl_features"].shape == (2, 2, 3)
    assert out["carl_teacher_frames"].tolist() == [2, 1]
