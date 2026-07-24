#!/usr/bin/env python
"""Sweep the WHAT-side blank penalty: how much voice do off-peak frames get?

Production Span-KD applies a blank penalty (delta=6) only on the WHERE path (the
FB occupancy). The WHAT path averages the RAW teacher posterior, so a shoulder
frame that is 99% blank contributes ~1% of its weight and is effectively erased —
which is why the pooled target comes out one-hot.

This sweeps a SECOND delta applied to the teacher posterior BEFORE the
occupancy-weighted average. delta_what=0 is production; delta_what -> inf is
"drop blank per frame and renormalise". Reports, per delta_what:

  q(top1)        pooled target's top-1 probability (production: ~one-hot)
  n_eff_vocab    exp(entropy) = effective number of candidates in the target
  top1==GT       does the target still name the ground-truth token
  off-peak w     share of the averaging weight held by non-argmax frames, after
                 the frame's own non-blank mass is taken into account
  rank-2         most frequent runner-up tokens, to judge signal vs noise

No training. Decides whether the WHAT axis has anything in it at all.
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np
import soundfile as sf
import torch

ROOT = "/workspace/transition_aware_kd"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_span_kd_targets import blank_penalty_logprobs, run_teacher  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward        # noqa: E402


def load_audio(path, sr_target):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == sr_target
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/train_clean_100.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "tokenizer_1024"))
    ap.add_argument("--delta-where", type=float, default=6.0)
    ap.add_argument("--delta-what", default="0,1,2,3,6,100")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    deltas = [float(d) for d in args.delta_what.split(",")]

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tok = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()][:args.limit]
    acc = {d: dict(top1=[], nvocab=[], eq=0, n=0, offpeak=[], r2=Counter()) for d in deltas}

    for i, r in enumerate(rows):
        wav = load_audio(r["audio_filepath"], sr)
        ids = [int(x) for x in tok.text_to_ids(r["text"]) if int(x) != blank]
        if not ids:
            continue
        lp = run_teacher(teacher, wav, args.device)                       # (T,V) sharp
        where = blank_penalty_logprobs(lp, blank, args.delta_where)
        gamma = ctc_forward_backward(where, ids, blank)["token_gamma"]     # (T,N)

        for d in deltas:
            what_lp = blank_penalty_logprobs(lp, blank, d)
            probs = np.exp(what_lp)
            nonblank_mass = 1.0 - probs[:, blank]                          # (T,)
            for u, y in enumerate(ids):
                w = gamma[:, u]
                s = w.sum()
                if s <= 1e-8:
                    continue
                wn = w / s
                # how much of the averaging weight sits off the peak frame,
                # after each frame's own non-blank mass is accounted for
                eff = wn * nonblank_mass
                if eff.sum() <= 1e-12:
                    continue
                eff = eff / eff.sum()
                acc[d]["offpeak"].append(float(1.0 - eff[int(np.argmax(wn))]))

                q = (wn[:, None] * probs).sum(0)
                q[blank] = 0.0
                q = q / max(float(q.sum()), 1e-12)
                order = np.argsort(q)[::-1]
                acc[d]["top1"].append(float(q[order[0]]))
                p = q[q > 1e-12]
                acc[d]["nvocab"].append(float(np.exp(-(p * np.log(p)).sum())))
                acc[d]["n"] += 1
                if int(order[0]) == y:
                    acc[d]["eq"] += 1
                    acc[d]["r2"][tok.ids_to_tokens([int(order[1])])[0]] += 1
        if (i + 1) % 50 == 0:
            print(f"  ..{i+1}/{len(rows)}", flush=True)

    print(f"\nWHAT-side blank penalty sweep — {len(rows)} utts, delta_where={args.delta_where}")
    hdr = (f"{'delta_what':>11}{'q(top1) mean':>14}{'median':>9}{'n_eff vocab':>13}"
           f"{'top1==GT':>10}{'q>0.9':>8}{'off-peak w':>12}")
    print(hdr); print("-" * len(hdr))
    out = {}
    for d in deltas:
        a = acc[d]
        t1 = np.array(a["top1"]); nv = np.array(a["nvocab"]); op = np.array(a["offpeak"])
        label = "inf" if d >= 50 else f"{d:g}"
        print(f"{label:>11}{t1.mean():>14.3f}{np.median(t1):>9.3f}{nv.mean():>13.2f}"
              f"{100*a['eq']/max(a['n'],1):>9.2f}%{100*(t1>0.9).mean():>7.1f}%{op.mean():>12.3f}")
        out[label] = dict(top1_mean=float(t1.mean()), top1_median=float(np.median(t1)),
                          n_eff_vocab=float(nv.mean()), top1_eq_gt=100*a["eq"]/max(a["n"],1),
                          frac_gt_09=float(100*(t1>0.9).mean()), offpeak_w=float(op.mean()))
    print("\nmost frequent runner-up token (spans where top-1 == GT):")
    for d in deltas:
        label = "inf" if d >= 50 else f"{d:g}"
        top = ", ".join(f"{k!r}x{v}" for k, v in acc[d]["r2"].most_common(6))
        print(f"  delta={label:>4} : {top}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
