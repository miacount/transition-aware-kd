#!/usr/bin/env python3
"""Validity unit test for the span-KD content/emission decomposition.

Checks on synthetic data that the edited _span_kd_loss:
  (1) beta=1  reproduces the ORIGINAL un-normalized span CE (bit-close), and
  (2) beta=0  equals the pure content term  -sum_v q(v) log s_bar(v).
Run BEFORE launching any K x beta sweep.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel  # noqa: E402


def make_batch(seed=0, B=2, T=10, K=12, N=3, top_k=4):
    g = torch.Generator().manual_seed(seed)
    log_probs = torch.log_softmax(torch.randn(B, T, K, generator=g, dtype=torch.float64), dim=-1)
    enc_len = torch.full((B,), T, dtype=torch.long)
    teacher_frames = torch.full((B,), T, dtype=torch.long)
    num_tokens = torch.full((B,), N, dtype=torch.long)
    support = torch.rand(B, N, T, generator=g, dtype=torch.float64) + 1e-3  # non-empty
    avg_ids = torch.stack([
        torch.stack([torch.randperm(K, generator=g)[:top_k] for _ in range(N)])
        for _ in range(B)
    ]).long()
    ap = torch.rand(B, N, top_k, generator=g, dtype=torch.float64)
    avg_probs = ap / ap.sum(-1, keepdim=True)  # normalized over top-k (matches builder)
    gates = torch.full((B, N), 0.9, dtype=torch.float64)  # all pass the 0.3 gate
    return dict(log_probs=log_probs, enc_len=enc_len, support=support, avg_ids=avg_ids,
               avg_probs=avg_probs, gates=gates, num_tokens=num_tokens,
               teacher_frames=teacher_frames)


def ref_loss(batch, uniform=False, content_only=False):
    """Independent reimplementation. content_only=False, uniform=False, is the
    ORIGINAL pre-decomposition CE: -sum q log s_u (un-normalized student)."""
    lp, enc_len = batch["log_probs"], batch["enc_len"]
    support, avg_ids, avg_probs = batch["support"], batch["avg_ids"], batch["avg_probs"]
    gates, num_tokens, tfr = batch["gates"], batch["num_tokens"], batch["teacher_frames"]
    B, T_s, K = lp.shape
    sp = lp.exp()
    total = torch.zeros((), dtype=torch.float64)
    total_w = torch.zeros((), dtype=torch.float64)
    for b in range(B):
        n, ts, tt = int(num_tokens[b]), int(enc_len[b]), int(tfr[b])
        sup_t = support[b, :n, :tt]
        base = torch.arange(ts, dtype=torch.float64) / max(ts, 1) * tt
        ia = torch.floor(base + 0.25 * tt / max(ts, 1)).long().clamp(0, tt - 1)
        ib = torch.floor(base + 0.75 * tt / max(ts, 1)).long().clamp(0, tt - 1)
        sup_s = 0.5 * (sup_t[:, ia] + sup_t[:, ib])
        sup_sum = sup_s.sum(1)
        sup_s = sup_s / sup_sum.clamp_min(1e-8).unsqueeze(1)
        s_avg = torch.matmul(sup_s, sp[b, :ts])
        ids = avg_ids[b, :n].clamp(0, K - 1)
        probs = avg_probs[b, :n].clone()
        if uniform:
            probs = torch.full_like(probs, 1.0 / probs.shape[1])
        sel = torch.gather(s_avg, 1, ids).clamp_min(1e-9)
        if content_only:
            sbar = sel / sel.sum(1, keepdim=True).clamp_min(1e-9)
            ce = -(probs * sbar.clamp_min(1e-9).log()).sum(1)
        else:
            ce = -(probs * sel.log()).sum(1)  # original
        w = gates[b, :n]
        active = (w >= 0.3) & (sup_sum > 1e-6)
        total = total + (ce[active] * w[active]).sum()
        total_w = total_w + w[active].sum()
    return total / total_w.clamp_min(1.0)


def call(batch, beta, uniform=False):
    stub = types.SimpleNamespace(
        span_kd_gate_threshold=0.3, span_kd_emit_beta=beta,
        span_kd_uniform_target=uniform,
    )
    return TransitionKDModel._span_kd_loss(
        stub, batch["log_probs"], batch["enc_len"], batch["support"],
        batch["avg_ids"], batch["avg_probs"], batch["gates"],
        batch["num_tokens"], batch["teacher_frames"],
    )


def main():
    ok = True
    for seed in range(5):
        b = make_batch(seed)
        # (1) beta=1 == original CE
        got, ref = call(b, 1.0), ref_loss(b)
        d1 = (got - ref).abs().item()
        # (2) beta=0 == content-only
        got0, ref0 = call(b, 0.0), ref_loss(b, content_only=True)
        d2 = (got0 - ref0).abs().item()
        # (3) uniform target beta=1 == original CE with flat probs
        gotu, refu = call(b, 1.0, uniform=True), ref_loss(b, uniform=True)
        d3 = (gotu - refu).abs().item()
        print(f"seed {seed}: beta1-vs-orig={d1:.2e}  beta0-vs-content={d2:.2e}  "
              f"uniform-vs-orig={d3:.2e}")
        ok = ok and d1 < 1e-9 and d2 < 1e-9 and d3 < 1e-9
    print("\nPASS ✅" if ok else "\nFAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
