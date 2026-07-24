#!/usr/bin/env python
"""Method-5 gate: does an occupancy gap exist between teacher and the span-KD
student, and does it predict token errors?

Per transcript token u, compute transcript-constrained FB occupancy (δ=6) for
teacher and student, normalize over time, and measure:
  - JS(teacher_occ_u, student_occ_u)       [the gap L_occ would close]
  - centroid offset |E_t teacher - E_t student|  (frames)
  - local correctness: student argmax == u at teacher occ-peak frame

NOISE FLOOR: same JS between two same-width (δ=6) students (seed diff). If
teacher-student JS ~= student-student JS, the gap is seed noise -> KILL.

Decision:  JS ~ floor, OR JS uncorrelated with error  -> no room, KILL L_occ.
           JS >> floor AND monotone with error         -> probe L1/JS occ loss.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_student import load_model, make_split_cfgs   # noqa: E402
from ctc_fb import batched_ctc_token_gamma                  # noqa: E402
import nemo.collections.asr as nemo_asr                     # noqa: E402


def js_div(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 1e-12
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)   # nats, [0, ln2]


def resample(v, L):
    if len(v) == L:
        return v
    xs = np.linspace(0, 1, len(v)); xt = np.linspace(0, 1, L)
    return np.interp(xt, xs, v)


@torch.no_grad()
def gammas_for_batch(model, batch, blank_id, bp):
    lp, enc_len, _ = model.forward(input_signal=batch["wavs"],
                                   input_signal_length=batch["wav_lens"])
    g = batched_ctc_token_gamma(lp, enc_len, batch["tokens"], batch["token_lens"],
                                blank_id, blank_penalty=bp)         # (B,N,T)
    return g, lp.argmax(-1), enc_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--student", required=True)       # span-KD d6
    ap.add_argument("--student2", required=True)      # d6 seed-diff (noise floor)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"[load] teacher {args.teacher}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(device).eval()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    print(f"[load] student  {args.student}")
    stu, cfg = load_model(args.config, args.student, device)
    print(f"[load] student2 {args.student2}")
    stu2, _ = load_model(args.config, args.student2, device)

    js_ts, js_ss, offs, corr = [], [], [], []
    for ds in make_split_cfgs(cfg, args.manifest):
        dl = stu._make_dataloader_from_cfg(ds, shuffle=False)
        for batch in dl:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            tg_b, _,     tlen = gammas_for_batch(teacher, batch, blank, args.delta)
            sg_b, sam_b, slen = gammas_for_batch(stu,     batch, blank, args.delta)
            s2g_b, _,    _    = gammas_for_batch(stu2,    batch, blank, args.delta)
            B = sg_b.shape[0]
            for b in range(B):
                Ts = int(slen[b].item()); Tt = int(tlen[b].item())
                n = int(batch["token_lens"][b].item())
                stok = batch["tokens"][b, :n].cpu().numpy()
                sam = sam_b[b, :Ts].cpu().numpy()
                for u in range(n):
                    tv = resample(tg_b[b, u, :Tt].cpu().numpy(), Ts)
                    sv = sg_b[b, u, :Ts].cpu().numpy()
                    s2v = s2g_b[b, u, :Ts].cpu().numpy()
                    if tv.sum() < 1e-6 or sv.sum() < 1e-6:
                        continue
                    js_ts.append(js_div(tv, sv))
                    if s2v.sum() > 1e-6:
                        js_ss.append(js_div(sv, s2v))
                    ax = np.arange(Ts)
                    ct = (tv * ax).sum() / tv.sum(); cs = (sv * ax).sum() / sv.sum()
                    offs.append(abs(ct - cs))
                    tpk = int(np.argmax(tv))
                    corr.append(int(sam[min(tpk, Ts - 1)] == stok[u]))

    js_ts = np.array(js_ts); js_ss = np.array(js_ss)
    offs = np.array(offs); corr = np.array(corr)
    L2 = np.log(2)
    print("\n" + "=" * 64)
    print(f"tokens analyzed            : {len(js_ts)}")
    print(f"JS teacher-student  (nats) : median {np.median(js_ts):.4f}  mean {js_ts.mean():.4f}  "
          f"(/ln2 = {js_ts.mean()/L2:.3f})")
    print(f"JS student-student  (nats) : median {np.median(js_ss):.4f}  mean {js_ss.mean():.4f}   <- NOISE FLOOR")
    print(f"JS ratio  (t-s / s-s)      : {js_ts.mean()/max(js_ss.mean(),1e-9):.2f}x   "
          f"(~1x => gap is seed noise)")
    print(f"centroid offset  (frames)  : median {np.median(offs):.2f}  mean {offs.mean():.2f}")
    print(f"local token correctness    : {100*corr.mean():.1f}%")
    # correlation JS vs error (bucket by JS quartile)
    print("\n--- JS(teacher-student) quartile  ->  token error rate ---")
    qs = np.quantile(js_ts, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (js_ts >= qs[i]) & (js_ts <= qs[i+1] if i == 3 else js_ts < qs[i+1])
        er = 100 * (1 - corr[m].mean()) if m.any() else 0
        print(f"  Q{i+1}  JS[{qs[i]:.3f},{qs[i+1]:.3f}]  err={er:5.1f}%   n={int(m.sum())}")
    from scipy.stats import spearmanr
    rho, p = spearmanr(js_ts, 1 - corr)
    print(f"\nSpearman JS vs error : rho={rho:.3f}  (p={p:.1e})")
    print("=" * 64)
    print("READ: JS~floor OR flat quartiles/rho~0 -> KILL L_occ.  "
          "JS>>floor AND monotone rho>0 -> probe.")


if __name__ == "__main__":
    main()
