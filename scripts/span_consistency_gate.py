#!/usr/bin/env python
"""Joint gate for two candidate on-axis levers, measured on the current d6 student.

(A) Support-Consistent Span-KD (Method-1):
    two dropout views -> pool each over teacher δ=6 support -> per-span JS(s1,s2).
    Compare to per-frame JS at the token peak. If span JS ~ 0 << frame JS, pooling
    washed out the variance -> span-consistency is a no-op (generic CR-CTC only).
    Also: does span-view JS concentrate on error spans?

(B) Span-content margin (the residual "right place, wrong token"):
    clean student argmax at the teacher occ-peak -> {correct, blank, other}.
    For 'other', is the wrong token inside the teacher top-k SET for that span?
    other-in-set high -> a discriminative margin on the set can fix it.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_student import load_model, make_split_cfgs   # noqa: E402
from ctc_fb import batched_ctc_token_gamma                  # noqa: E402
import nemo.collections.asr as nemo_asr                     # noqa: E402


def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)) or "Dropout" in type(m).__name__:
            m.train()


def js(p, q, drop_blank=None):
    p = p.copy(); q = q.copy()
    if drop_blank is not None:
        p[drop_blank] = 0; q[drop_blank] = 0
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        k = a > 1e-12
        return float(np.sum(a[k] * np.log(a[k] / np.clip(b[k], 1e-12, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def fwd(model, batch):
    lp, enc_len, _ = model.forward(input_signal=batch["wavs"],
                                   input_signal_length=batch["wav_lens"])
    return lp, enc_len


def pool(support_resamp, prob):
    # support_resamp (N,T) normalized rows, prob (T,V) -> (N,V)
    return support_resamp @ prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(dev).eval()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    stu, cfg = load_model(args.config, args.student, dev)

    spanJS, spanJS_tok, frameJS, err, inset_other = [], [], [], [], []
    n_correct = n_blank = n_other = 0

    for ds in make_split_cfgs(cfg, args.manifest):
        dl = stu._make_dataloader_from_cfg(ds, shuffle=False)
        for batch in dl:
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            # teacher occupancy support (δ=6) + teacher logprob
            t_lp, t_len = fwd(teacher, batch)
            supp = batched_ctc_token_gamma(t_lp, t_len, batch["tokens"], batch["token_lens"],
                                           blank, blank_penalty=args.delta)   # (B,N,Tt)
            t_prob = t_lp.exp()
            # clean student view
            stu.eval()
            s_lp0, s_len = fwd(stu, batch)
            s_prob0 = s_lp0.exp()
            s_arg0 = s_lp0.argmax(-1)
            # two dropout views
            enable_dropout(stu)
            s_lp1, _ = fwd(stu, batch); p1 = s_lp1.exp()
            s_lp2, _ = fwd(stu, batch); p2 = s_lp2.exp()
            stu.eval()

            B = s_lp0.shape[0]
            for b in range(B):
                Tt = int(t_len[b].item()); Ts = int(s_len[b].item())
                n = int(batch["token_lens"][b].item())
                toks = batch["tokens"][b, :n]
                sup_t = supp[b, :n, :Tt]                                   # (n,Tt) teacher grid
                # resample support to student grid
                idx = (torch.arange(Ts, device=dev).float() / max(Ts, 1) * Tt).long().clamp(0, Tt-1)
                sup_s = sup_t[:, idx]                                      # (n,Ts)
                rs = sup_s.sum(1, keepdim=True).clamp_min(1e-8)
                sup_s_n = sup_s / rs
                # teacher q (teacher grid pooling)
                rt = sup_t.sum(1, keepdim=True).clamp_min(1e-8)
                q = pool(sup_t / rt, t_prob[b, :Tt])                      # (n,V)
                # student pooled views
                s0 = pool(sup_s_n, s_prob0[b, :Ts])
                s1 = pool(sup_s_n, p1[b, :Ts]); s2 = pool(sup_s_n, p2[b, :Ts])
                q = q.cpu().numpy(); s0n = s0.cpu().numpy()
                s1n = s1.cpu().numpy(); s2n = s2.cpu().numpy()
                for u in range(n):
                    if sup_s[u].sum() < 1e-6:
                        continue
                    tok = int(toks[u].item())
                    # teacher top-k SET (drop blank)
                    qv = q[u].copy(); qv[blank] = 0
                    topk = set(np.argpartition(qv, -args.top_k)[-args.top_k:].tolist())
                    # (A) span-view JS (token semantics = blank-dropped)
                    jsp = js(s1n[u], s2n[u], drop_blank=blank)
                    spanJS.append(jsp)
                    # teacher occ-peak frame -> student grid
                    tpk = int(torch.argmax(sup_t[u]).item())
                    spk = min(int(round(tpk * Ts / max(Tt, 1))), Ts - 1)
                    # (A') per-frame JS at peak (blank-dropped)
                    f1 = p1[b, spk].cpu().numpy(); f2 = p2[b, spk].cpu().numpy()
                    frameJS.append(js(f1, f2, drop_blank=blank))
                    # (B) argmax decomposition at peak (clean view)
                    a = int(s_arg0[b, spk].item())
                    if a == tok:
                        n_correct += 1; err.append(0)
                    elif a == blank:
                        n_blank += 1; err.append(1)
                    else:
                        n_other += 1; err.append(1)
                        inset_other.append(int(a in topk))
                    spanJS_tok.append(jsp)

    spanJS = np.array(spanJS); frameJS = np.array(frameJS); err = np.array(err)
    L2 = np.log(2); N = len(spanJS)
    tot = n_correct + n_blank + n_other
    print("\n" + "=" * 66)
    print(f"spans analyzed : {N}")
    print("\n--- (A) Support-Consistent: view disagreement (blank-dropped JS, nats) ---")
    print(f"  span-pooled  JS(s1,s2) : median {np.median(spanJS):.4f}  mean {spanJS.mean():.4f}")
    print(f"  frame@peak   JS(p1,p2) : median {np.median(frameJS):.4f}  mean {frameJS.mean():.4f}")
    print(f"  span/frame ratio       : {spanJS.mean()/max(frameJS.mean(),1e-9):.2f}  "
          f"(<<1 => pooling washed variance => span-cons ~ no-op)")
    # does span JS concentrate on error spans?
    qs = np.quantile(spanJS, [0,.25,.5,.75,1.0])
    print("  span-JS quartile -> error rate:")
    for i in range(4):
        m = (spanJS>=qs[i]) & (spanJS<=qs[i+1] if i==3 else spanJS<qs[i+1])
        print(f"    Q{i+1} JS[{qs[i]:.3f},{qs[i+1]:.3f}]  err={100*err[m].mean():.1f}%  n={int(m.sum())}")
    from scipy.stats import spearmanr
    rho,_ = spearmanr(spanJS, err)
    print(f"  Spearman(span-JS, error): rho={rho:.3f}")
    print("\n--- (B) residual decomposition @ teacher occ-peak (clean view) ---")
    print(f"  correct : {100*n_correct/tot:.1f}%")
    print(f"  blank   : {100*n_blank/tot:.1f}%   (emission/timing lever)")
    print(f"  other   : {100*n_other/tot:.1f}%   (substitution lever)")
    if inset_other:
        io = np.array(inset_other)
        print(f"    of 'other': {100*io.mean():.1f}% are INSIDE teacher top-{args.top_k} SET "
              f"-> margin can reach them")
    print("=" * 66)
    print("READ  A: span/frame<<1 or flat quartiles -> Method-1 likely generic-CR-CTC only.")
    print("      B: 'other' dominant & mostly in-set -> margin lever real; "
          "'blank' dominant -> emission-sharpen lever.")


if __name__ == "__main__":
    main()
