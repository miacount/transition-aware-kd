#!/usr/bin/env python3
"""Numerical validity gate for _boundary_kd_loss (all 4 cells).

Builds a tiny synthetic batch and checks each cell's loss against an independent
pure-numpy reference. Guards the properties the method depends on:
  - full-softmax CE with NO {blank,y_u} renormalisation (shortcut-proof)
  - RAW residual weighting, batch-normalised denominator (sum r over the batch)
  - cell 2/3 differ only by m_delta vs m_zero
  - cell 4 preserves per-token residual mass, excludes the spike, uniform on dist=1
  - cell 1 gathers the soft posterior at the right residual frame
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel  # noqa: E402


class Stub:
    """Only the attributes _boundary_kd_loss reads."""
    def __init__(self, cell, blank_id, V, train_min=0.0):
        self.boundary_kd_cell = cell
        self.boundary_kd_train_min = train_min
        self.blank_id = blank_id
    loss = TransitionKDModel._boundary_kd_loss


def make_batch(V, blank):
    # 1 utterance, ts=tt=6 frames, n=2 tokens y=[3,5]
    torch.manual_seed(0)
    logits = torch.randn(1, 6, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    # residual entries: (t=1,u=0,r=0.3), (t=3,u=0,r=0.1), (t=4,u=1,r=0.2)
    R = 3
    b = {
        "bd_res_t": torch.tensor([[1, 3, 4]]),
        "bd_res_u": torch.tensor([[0, 0, 1]]),
        "bd_res_r": torch.tensor([[0.3, 0.1, 0.2]]),
        "bd_res_valid": torch.tensor([[True, True, True]]),
        "bd_m_delta": torch.tensor([[0.1, 0.5, 0.2, 0.6, 0.7, 0.3]]),
        "bd_m_zero":  torch.tensor([[0.05, 0.2, 0.1, 0.25, 0.3, 0.15]]),
        "bd_spike":   torch.tensor([[2, 4]]),         # token0 spike@2, token1 spike@4
        "bd_y":       torch.tensor([[3, 5]]),
        "bd_num_tokens": torch.tensor([2]),
        "bd_teacher_frames": torch.tensor([6]),
    }
    enc_len = torch.tensor([6])
    return log_probs, enc_len, b


def ref_hard(log_probs, entries, m_lookup, y, blank):
    """entries: list of (t_teacher, u, weight). ts==tt so fs=t."""
    lp = log_probs[0].numpy()
    num = 0.0; den = 0.0
    for t, u, w in entries:
        m = m_lookup[t]
        ce = -((1 - m) * lp[t, blank] + m * lp[t, y[u]])
        num += w * ce; den += w
    return num / den


def main():
    V, blank = 8, 7
    log_probs, enc_len, b = make_batch(V, blank)
    md = b["bd_m_delta"][0].numpy(); mz = b["bd_m_zero"][0].numpy()
    y = b["bd_y"][0].numpy()
    ok = True

    # ---- cell 2: hard, m_delta, raw residual weights ----
    got = float(Stub(2, blank, V).loss(log_probs, enc_len, b))
    ref = ref_hard(log_probs, [(1, 0, 0.3), (3, 0, 0.1), (4, 1, 0.2)], md, y, blank)
    print(f"[cell2] got {got:.6f}  ref {ref:.6f}  {'OK' if abs(got-ref)<1e-5 else 'FAIL'}")
    ok &= abs(got - ref) < 1e-5

    # ---- cell 3: same entries, m_zero ----
    got = float(Stub(3, blank, V).loss(log_probs, enc_len, b))
    ref = ref_hard(log_probs, [(1, 0, 0.3), (3, 0, 0.1), (4, 1, 0.2)], mz, y, blank)
    print(f"[cell3] got {got:.6f}  ref {ref:.6f}  {'OK' if abs(got-ref)<1e-5 else 'FAIL'}")
    ok &= abs(got - ref) < 1e-5

    # ---- cell 4: mass A_u redistributed to dist=1, spike excluded, m_delta ----
    # token0: A=0.4, spike@2 -> shoulders {1,3}, weight 0.2 each
    # token1: A=0.2, spike@4 -> shoulders {3,5}, weight 0.1 each
    got = float(Stub(4, blank, V).loss(log_probs, enc_len, b))
    ref = ref_hard(log_probs,
                   [(1, 0, 0.2), (3, 0, 0.2), (3, 1, 0.1), (5, 1, 0.1)], md, y, blank)
    print(f"[cell4] got {got:.6f}  ref {ref:.6f}  {'OK' if abs(got-ref)<1e-5 else 'FAIL'}")
    ok &= abs(got - ref) < 1e-5
    # cell4 total weight must equal total residual mass A (0.4+0.2=0.6)
    st = Stub(4, blank, V); _ = st.loss(log_probs, enc_len, b)
    wsum = float(st._boundary_diag["weight_sum"])
    print(f"[cell4] weight_sum {wsum:.4f} == total residual mass 0.6  "
          f"{'OK' if abs(wsum-0.6)<1e-5 else 'FAIL'}")
    ok &= abs(wsum - 0.6) < 1e-5

    # ---- cell 1: soft, full-softmax CE, gathered at residual frames ----
    F = 3  # residual frames unique: {1,3,4}
    soft_p = torch.softmax(torch.randn(1, F, V), dim=-1)
    b1 = dict(b)
    b1["bd_soft_t"] = torch.tensor([[1, 3, 4]])
    b1["bd_soft_p"] = soft_p
    got = float(Stub(1, blank, V).loss(log_probs, enc_len, b1))
    lp = log_probs[0].numpy(); sp = soft_p[0].numpy()
    ent = [(1, 0, 0.3), (3, 1, 0.1), (4, 2, 0.2)]  # frame->soft row idx, weight
    num = sum(w * (-(sp[fi] * lp[t]).sum()) for (t, fi, w) in ent)
    den = 0.3 + 0.1 + 0.2
    ref = num / den
    print(f"[cell1] got {got:.6f}  ref {ref:.6f}  {'OK' if abs(got-ref)<1e-5 else 'FAIL'}")
    ok &= abs(got - ref) < 1e-5

    # ---- shortcut-proof: putting all student mass on a THIRD token must NOT
    #      drive cell2 loss to ~0 (full-softmax, no renorm) ----
    lp_bad = torch.full((1, 6, V), -10.0)
    lp_bad[..., 0] = 0.0  # token 0 gets ~all mass; y=[3,5], blank=7
    lp_bad = torch.log_softmax(lp_bad, dim=-1)
    bad = float(Stub(2, blank, V).loss(lp_bad, enc_len, b))
    print(f"[shortcut] third-token dodge loss {bad:.3f} (must be large, >1) "
          f"{'OK' if bad > 1.0 else 'FAIL'}")
    ok &= bad > 1.0

    # ---- runtime mask: train_min above a residual drops that entry ----
    got = float(Stub(2, blank, V, train_min=0.15).loss(log_probs, enc_len, b))
    ref = ref_hard(log_probs, [(1, 0, 0.3), (4, 1, 0.2)], md, y, blank)  # 0.1 dropped
    print(f"[mask]  train_min=0.15 got {got:.6f}  ref {ref:.6f}  "
          f"{'OK' if abs(got-ref)<1e-5 else 'FAIL'}")
    ok &= abs(got - ref) < 1e-5

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
