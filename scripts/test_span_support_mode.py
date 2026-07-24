#!/usr/bin/env python3
"""Validity gate for --span-support-mode (time-axis occupancy-shape ablation).

Checks, on synthetic and on REAL delta=6 targets:
  1. mode=gamma is a bit-exact no-op  -> the anchor cell reproduces the current
     method, so any delta in the suite is attributable to the knob.
  2. uniform keeps exactly the frames above eps*peak, and is flat on them.
  3. rect(width=w) keeps exactly w frames centred on argmax (clipped at edges),
     and rect(width=1) keeps only the teacher's spike frame.
  4. every mode leaves at least one frame per token -> no span is silently
     dropped by the sup_sum > 1e-6 gate (which would change the token set the
     loss averages over and confound the comparison).
Run before launching the suite.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel  # noqa: E402


class _Stub:
    """Minimal stand-in: _transform_support only reads these three attrs."""

    def __init__(self, mode, eps=0.01, width=0):
        self.span_kd_support_mode = mode
        self.span_kd_support_eps = eps
        self.span_kd_support_width = width

    transform = TransitionKDModel._transform_support


def main():
    ok = True

    # ---- synthetic --------------------------------------------------------
    # token 0: sharp spike + shoulder;  token 1: flat-ish triple
    sup = torch.tensor([
        [0.0, 0.02, 0.90, 0.08, 0.0, 0.0],
        [0.0, 0.0, 0.30, 0.40, 0.30, 0.0],
    ])

    g = _Stub("gamma").transform(sup)
    same = torch.equal(g, sup)
    print(f"[1] gamma is a no-op                     : {same}")
    ok &= same

    u = _Stub("uniform", eps=0.01).transform(sup)
    exp_u = torch.tensor([[0., 1., 1., 1., 0., 0.], [0., 0., 1., 1., 1., 0.]])
    good = torch.equal(u, exp_u)
    print(f"[2] uniform = flat over frames > eps*peak: {good}  widths={u.sum(1).tolist()}")
    ok &= good

    r1 = _Stub("rect", width=1).transform(sup)
    exp_r1 = torch.tensor([[0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]])
    good = torch.equal(r1, exp_r1)
    print(f"[3] rect(w=1) = argmax frame only        : {good}")
    ok &= good

    r3 = _Stub("rect", width=3).transform(sup)
    good = torch.equal(r3.sum(1), torch.tensor([3., 3.])) and r3[0, 1] == 1 and r3[0, 3] == 1
    print(f"[4] rect(w=3) = 3 frames around argmax   : {good}  widths={r3.sum(1).tolist()}")
    ok &= good

    # edge clipping: spike at frame 0 -> window truncated, must stay non-empty
    edge = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
    re = _Stub("rect", width=5).transform(edge)
    good = re.sum() >= 1
    print(f"[5] rect clips at utterance edge, non-empty: {bool(good)}  width={re.sum().item()}")
    ok &= bool(good)

    # ---- real targets -----------------------------------------------------
    man = "data/train_clean_100.teacher_only_d6.json"
    if os.path.exists(man):
        import json
        rows = [json.loads(l) for l in open(man) if l.strip()][:150]
        stats = {}
        empty = {}
        for name, stub in [("gamma", _Stub("gamma")),
                           ("uniform", _Stub("uniform", eps=0.01)),
                           ("rect_w1", _Stub("rect", width=1)),
                           ("rect_w3", _Stub("rect", width=3))]:
            tot, n, emp = 0.0, 0, 0
            for r in rows:
                s = torch.load(os.path.join("data", r["teacher_span_kd_path"]),
                               map_location="cpu", weights_only=False)["support"].float()
                keep = s.sum(1) > 1e-6
                if keep.sum() == 0:
                    continue
                out = stub.transform(s[keep])
                w = (out > 0).float().sum(1)
                tot += float(w.sum()); n += int(w.numel())
                emp += int((out.sum(1) <= 1e-6).sum())
            stats[name] = tot / max(n, 1)
            empty[name] = emp
        print("\nreal delta=6 targets (150 utts), mean frames kept per token:")
        for k, v in stats.items():
            print(f"    {k:<9} {v:5.2f}   empty spans: {empty[k]}")
        no_empty = all(v == 0 for v in empty.values())
        print(f"[6] no mode empties a span               : {no_empty}")
        ok &= no_empty
    else:
        print(f"\n[skip] {man} not found — synthetic checks only")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
