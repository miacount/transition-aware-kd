import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_span_kd_targets import teacher_nontarget_topm  # noqa: E402
from model import TransitionKDModel  # noqa: E402


def test_teacher_topm_plus_tail_is_exact_and_excludes_gt_blank():
    # Vocabulary: GT=0, candidates=1..3, blank=4.
    probs = np.array([
        [0.70, 0.12, 0.08, 0.05, 0.05],
        [0.60, 0.18, 0.10, 0.07, 0.05],
    ], dtype=np.float64)
    ids, vals, tail = teacher_nontarget_topm(
        np.log(probs), np.ones((2, 1)), [0], blank_id=4, top_m=2)
    assert ids.shape == vals.shape == (1, 2)
    assert 0 not in ids[0] and 4 not in ids[0]
    np.testing.assert_allclose(vals.sum(1) + tail, 1.0, atol=1e-6)
    # Candidate ordering is preserved; class 1 then class 2.
    assert ids[0].tolist() == [1, 2]


def test_teacher_mass3_preserves_blank_gt_and_nontarget_mass():
    probs = np.array([
        [0.70, 0.12, 0.08, 0.05, 0.05],
        [0.60, 0.18, 0.10, 0.07, 0.05],
    ], dtype=np.float64)
    _, _, _, mass3 = teacher_nontarget_topm(
        np.log(probs), np.ones((2, 1)), [0], blank_id=4, top_m=2,
        return_mass3=True)
    # Uniform occupancy average: blank=.05, GT=.65, remaining classes=.30.
    np.testing.assert_allclose(mass3[0], [0.05, 0.65, 0.30], atol=1e-6)


class _Stub:
    blank_id = 4
    span_kd_gate_threshold = 0.3
    span_ntdk_reliability = False
    span_ntdk_mass_weighted = False
    span_ntdk_rel_min = 1e-4
    span_ntdk_rel_max = 5e-2
    span_ntdk_rel_gt_min = 0.5
    span_ntdk_rel_power = 0.5
    _span_ntdk_loss = TransitionKDModel._span_ntdk_loss

    @staticmethod
    def _transform_support(support):
        return support


def _call(student_probs, teacher_mass3=None, reliability=False, mass_weighted=False):
    model = _Stub()
    model.span_ntdk_reliability = reliability
    model.span_ntdk_mass_weighted = mass_weighted
    tm = None if teacher_mass3 is None else torch.tensor([[teacher_mass3]], dtype=torch.float32)
    lp = torch.tensor(student_probs, dtype=torch.float32).log().view(1, 1, 5)
    lp = lp.repeat(1, 2, 1).requires_grad_()
    loss = model._span_ntdk_loss(
        lp, torch.tensor([2]),
        support=torch.ones(1, 1, 2),
        dark_ids=torch.tensor([[[1, 2]]]),
        dark_probs=torch.tensor([[[1 / 3, 1 / 2]]]),
        dark_tail_prob=torch.tensor([[1 / 6]]),
        gates=torch.ones(1, 1),
        num_tokens=torch.tensor([1]),
        teacher_frames=torch.tensor([2]),
        tokens=torch.tensor([[0]]),
        token_lens=torch.tensor([1]),
        teacher_mass3=tm,
    )
    return loss, lp


def test_ntdk_zero_at_matching_conditional_distribution_and_has_gradient():
    # Removing GT=0 and blank=4 gives [1/3, 1/2, 1/6] over classes 1,2,tail(3).
    match, lp = _call([0.2, 0.2, 0.3, 0.1, 0.2])
    assert abs(float(match.detach())) < 2e-6
    match.backward()
    assert torch.isfinite(lp.grad).all()

    mismatch, _ = _call([0.2, 0.3, 0.1, 0.2, 0.2])
    assert float(mismatch.detach()) > 0.1


class _CoarseStub:
    blank_id = 4
    span_kd_gate_threshold = 0.3
    span_primary_mode = "mass3"
    _span_coarse_loss = TransitionKDModel._span_coarse_loss

    @staticmethod
    def _transform_support(support):
        return support


def _coarse_call(student_probs, teacher_mass3, mode="mass3"):
    model = _CoarseStub()
    model.span_primary_mode = mode
    lp = torch.tensor(student_probs, dtype=torch.float32).log().view(1, 1, 5)
    lp = lp.repeat(1, 2, 1).requires_grad_()
    loss = model._span_coarse_loss(
        lp, torch.tensor([2]), support=torch.ones(1, 1, 2),
        gates=torch.ones(1, 1), num_tokens=torch.tensor([1]),
        teacher_frames=torch.tensor([2]), tokens=torch.tensor([[0]]),
        token_lens=torch.tensor([1]),
        teacher_mass3=torch.tensor([[teacher_mass3]], dtype=torch.float32),
    )
    return loss, lp


def test_mass3_zero_at_matching_coarse_distribution_and_hard_gt_is_ce():
    match, lp = _coarse_call(
        [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.6])
    assert abs(float(match.detach())) < 2e-6
    match.backward()
    assert torch.isfinite(lp.grad).all()

    mismatch, _ = _coarse_call(
        [0.4, 0.1, 0.1, 0.1, 0.3], [0.2, 0.2, 0.6])
    assert float(mismatch.detach()) > 0.05

    hard_gt, _ = _coarse_call(
        [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.6], mode="hard_gt")
    torch.testing.assert_close(
        hard_gt.detach(), torch.tensor(-np.log(0.2), dtype=torch.float32))


def test_gt_nt_ignores_pooled_blank_mass_and_matches_conditional_distribution():
    # Teacher conditional is [GT=.25, NT=.75]. Student pooled blank differs
    # (.60 vs .20), but conditional content matches, so the loss must vanish.
    match, lp = _coarse_call(
        [0.1, 0.1, 0.1, 0.1, 0.6], [0.2, 0.2, 0.6], mode="gt_nt")
    assert abs(float(match.detach())) < 2e-6
    match.backward()
    assert torch.isfinite(lp.grad).all()

    mismatch, _ = _coarse_call(
        [0.3, 0.1, 0.1, 0.1, 0.4], [0.2, 0.2, 0.6], mode="gt_nt")
    assert float(mismatch.detach()) > 0.05


def test_ntdk_reliability_drops_teacher_wrong_occurrence():
    student = [0.2, 0.3, 0.1, 0.2, 0.2]
    kept, _ = _call(student, [0.1, 0.8, 0.1], reliability=True)
    assert float(kept.detach()) > 0.1

    dropped, _ = _call(student, [0.1, 0.1, 0.8], reliability=True)
    assert abs(float(dropped.detach())) < 1e-8


def test_ntdk_mass_weighted_keeps_token_denominator_and_scales_by_nt_mass():
    student = [0.2, 0.3, 0.1, 0.2, 0.2]
    plain, _ = _call(student)
    weighted, _ = _call(
        student, teacher_mass3=[0.1, 0.8, 0.1], mass_weighted=True)
    torch.testing.assert_close(weighted.detach(), 0.1 * plain.detach())


class _ShoulderStub:
    blank_id = 4
    span_kd_gate_threshold = 0.3
    _span_shoulder_loss = TransitionKDModel._span_shoulder_loss


def _shoulder_call(core, expanded, student_probs):
    model = _ShoulderStub()
    lp = torch.tensor(student_probs, dtype=torch.float32).log().view(1, 2, 5)
    lp.requires_grad_()
    loss = model._span_shoulder_loss(
        lp, torch.tensor([2]),
        torch.tensor([[core]], dtype=torch.float32),
        torch.tensor([[expanded]], dtype=torch.float32),
        torch.ones(1, 1), torch.tensor([1]), torch.tensor([2]),
        torch.tensor([[0]]), torch.tensor([1]))
    return loss, lp


def test_shoulder_loss_is_exactly_zero_when_delta_adds_no_support():
    loss, lp = _shoulder_call(
        [1.0, 0.0], [1.0, 0.0],
        [[0.7, 0.1, 0.05, 0.05, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.equal(lp.grad, torch.zeros_like(lp.grad))


def test_shoulder_loss_uses_only_positive_residual_frames():
    # Expanded support moves mass from frame 0 to frame 1. Only the positive
    # addition at frame 1 is supervised, so the loss is -log p_S(GT|frame 1).
    loss, lp = _shoulder_call(
        [1.0, 0.0], [0.5, 0.5],
        [[0.8, 0.05, 0.05, 0.05, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]])
    torch.testing.assert_close(
        loss.detach(), torch.tensor(-np.log(0.2), dtype=torch.float32))
    loss.backward()
    assert torch.isfinite(lp.grad).all()
