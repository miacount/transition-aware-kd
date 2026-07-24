#!/usr/bin/env python
"""CNW Step-1 diagnostic: does the implemented weight w_u = clip(r_u*d_u) put its
mass on spans where the TEACHER IS WRONG?

Implemented CNW uses teacher DECISIVENESS (top1-top2), never teacher CORRECTNESS.
Pathology: if the teacher is confidently wrong (top1 != GT), r_u is large AND the
student (correctly) puts little mass on that wrong token, so d_u is large too ->
w_u is MAXIMAL exactly where the teacher should be ignored.

Reports, per split: teacher span top-1 accuracy, mean w_u for teacher-correct vs
teacher-wrong spans, and the share of total KD weight mass going to wrong spans.
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
    ap.add_argument("--student", required=True, help="baseline d6 student")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--wmin", type=float, default=0.2)
    ap.add_argument("--wmax", type=float, default=5.0)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(dev).eval()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    stu, cfg = load_model(args.config, args.student, dev)

    for ds in make_split_cfgs(cfg, args.manifest):
        name = ds.get("name")
        W, CORR = [], []
        dl = stu._make_dataloader_from_cfg(ds, shuffle=False)
        for batch in dl:
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            t_lp, t_len, _ = teacher.forward(input_signal=batch["wavs"],
                                             input_signal_length=batch["wav_lens"])
            s_lp, s_len, _ = stu.forward(input_signal=batch["wavs"],
                                         input_signal_length=batch["wav_lens"])
            supp = batched_ctc_token_gamma(t_lp, t_len, batch["tokens"], batch["token_lens"],
                                           blank, blank_penalty=args.delta)
            t_prob = t_lp.exp(); s_prob = s_lp.exp()
            for b in range(supp.shape[0]):
                Tt = int(t_len[b].item()); ts = int(s_len[b].item())
                n = int(batch["token_lens"][b].item())
                gt = batch["tokens"][b, :n]
                sup_t = supp[b, :n, :Tt]
                if n == 0:
                    continue
                # teacher span-aggregated target q_u (blank dropped, top-k renorm) -- as built
                q = (sup_t / sup_t.sum(1, keepdim=True).clamp_min(1e-8)) @ t_prob[b, :Tt]
                q[:, blank] = 0.0
                q = q / q.sum(1, keepdim=True).clamp_min(1e-12)
                vals, ids = torch.topk(q, min(args.top_k, q.shape[1]), dim=1)
                probs = vals / vals.sum(1, keepdim=True).clamp_min(1e-12)
                r_u = (probs[:, 0] - probs[:, 1]).clamp_min(0.0)
                # student pooled prob on teacher top-1 (loss's pooling)
                base = torch.arange(ts, device=dev, dtype=torch.float32) / max(ts, 1) * Tt
                ia = torch.floor(base + 0.25 * Tt / max(ts, 1)).long().clamp(0, Tt - 1)
                ib = torch.floor(base + 0.75 * Tt / max(ts, 1)).long().clamp(0, Tt - 1)
                sup_s = 0.5 * (sup_t[:, ia] + sup_t[:, ib])
                ss = sup_s.sum(1)
                ok = ss > 1e-6
                if ok.sum() == 0:
                    continue
                sup_sn = sup_s[ok] / ss[ok].unsqueeze(1)
                s_avg = sup_sn @ s_prob[b, :ts]
                top1 = ids[ok, 0]
                s_on = s_avg.gather(1, top1.unsqueeze(1)).squeeze(1)
                d_u = (1.0 - s_on).clamp(0, 1)
                w = torch.clamp(r_u[ok].clamp_min(1e-6) ** args.alpha
                                * d_u.clamp_min(1e-6) ** args.gamma, args.wmin, args.wmax)
                w = w / w.mean().clamp_min(1e-8)          # per-utt mean-1 (gates~1)
                W.append(w.cpu().numpy())
                CORR.append((top1 == gt[ok]).cpu().numpy())

        w = np.concatenate(W); c = np.concatenate(CORR).astype(bool)
        acc = 100 * c.mean()
        print(f"\n===== {name}  (spans={len(w)}) =====")
        print(f"teacher span top-1 == GT : {acc:.1f}%   (wrong: {100-acc:.1f}%)")
        print(f"mean w_u  teacher-CORRECT : {w[c].mean():.3f}")
        print(f"mean w_u  teacher-WRONG   : {w[~c].mean():.3f}   "
              f"<-- ratio {w[~c].mean()/max(w[c].mean(),1e-9):.2f}x")
        share = 100 * w[~c].sum() / w.sum()
        print(f"share of TOTAL KD weight on teacher-WRONG spans : {share:.1f}%  "
              f"(their span share {100*(~c).mean():.1f}%)")
        qs = np.quantile(w, [0, .25, .5, .75, 1.0])
        print("  w_u quartile -> teacher top-1 accuracy:")
        for i in range(4):
            m = (w >= qs[i]) & (w <= qs[i+1] if i == 3 else w < qs[i+1])
            if m.any():
                print(f"    Q{i+1} w[{qs[i]:.2f},{qs[i+1]:.2f}]  teacher-acc={100*c[m].mean():5.1f}%  n={int(m.sum())}")


if __name__ == "__main__":
    main()
