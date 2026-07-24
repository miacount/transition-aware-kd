#!/usr/bin/env python
"""Method-1 gate 3.1: effective span width on the STUDENT grid.

n_eff(u) = 1 / sum_j a~_u(j)^2   where a~ is the teacher δ=6 support RESAMPLED to
the student frame grid and normalized -- EXACTLY as _span_kd_loss pools it.

n_eff ~ 1 : span pools ~one student frame -> span-consistency == frame-consistency
            (selected-frame CR-CTC) -> novelty weak, drop Method-1.
n_eff >=3 : genuine multi-frame span -> span-level consistency is a real object.
Also reports n_eff on the teacher grid, to show how much resampling collapsed it.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_student import load_model, make_split_cfgs
from ctc_fb import batched_ctc_token_gamma
import nemo.collections.asr as nemo_asr


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(dev).eval()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    stu, cfg = load_model(args.config, args.student, dev)

    neff_s, neff_t = [], []
    for ds in make_split_cfgs(cfg, args.manifest):
        dl = stu._make_dataloader_from_cfg(ds, shuffle=False)
        for batch in dl:
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            t_lp, t_len, _ = teacher.forward(input_signal=batch["wavs"],
                                             input_signal_length=batch["wav_lens"])
            _, s_len, _ = stu.forward(input_signal=batch["wavs"],
                                      input_signal_length=batch["wav_lens"])
            supp = batched_ctc_token_gamma(t_lp, t_len, batch["tokens"], batch["token_lens"],
                                           blank, blank_penalty=args.delta)   # (B,N,Tt)
            for b in range(supp.shape[0]):
                Tt = int(t_len[b].item()); ts = int(s_len[b].item())
                n = int(batch["token_lens"][b].item())
                sup_t = supp[b, :n, :Tt]                                      # (n,Tt)
                # teacher-grid n_eff
                st_sum = sup_t.sum(1, keepdim=True).clamp_min(1e-8)
                at = sup_t / st_sum
                ne_t = 1.0 / (at.pow(2).sum(1).clamp_min(1e-12))
                # resample to student grid EXACTLY like _span_kd_loss
                base = torch.arange(ts, device=dev, dtype=torch.float32) / max(ts, 1) * Tt
                idx_a = torch.floor(base + 0.25 * Tt / max(ts, 1)).long().clamp(0, Tt - 1)
                idx_b = torch.floor(base + 0.75 * Tt / max(ts, 1)).long().clamp(0, Tt - 1)
                sup_s = 0.5 * (sup_t[:, idx_a] + sup_t[:, idx_b])            # (n,ts)
                ss = sup_s.sum(1)
                valid = ss > 1e-6
                a_s = sup_s[valid] / ss[valid].unsqueeze(1)
                ne_s = 1.0 / (a_s.pow(2).sum(1).clamp_min(1e-12))
                neff_s.append(ne_s.cpu().numpy())
                neff_t.append(ne_t[valid].cpu().numpy())

    ns = np.concatenate(neff_s); nt = np.concatenate(neff_t)
    def frac(a, lo, hi):
        return 100 * np.mean((a >= lo) & (a < hi))
    print("\n" + "=" * 60)
    print(f"tokens: {len(ns)}   (teacher δ={args.delta} support)")
    print(f"n_eff  TEACHER grid : median {np.median(nt):.2f}  mean {nt.mean():.2f}")
    print(f"n_eff  STUDENT grid : median {np.median(ns):.2f}  mean {ns.mean():.2f}")
    print("\nstudent-grid n_eff distribution:")
    print(f"  ~one-frame  n_eff<1.25 : {frac(ns,0,1.25):5.1f}%")
    print(f"  [1.25,2)               : {frac(ns,1.25,2):5.1f}%")
    print(f"  [2,3)                  : {frac(ns,2,3):5.1f}%")
    print(f"  >=3 (real span)        : {100*np.mean(ns>=3):5.1f}%")
    print("=" * 60)
    print("READ: mostly n_eff~1 -> span-consistency == frame-consistency -> DROP Method-1.")


if __name__ == "__main__":
    main()
