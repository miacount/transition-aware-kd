#!/usr/bin/env python3
"""Trace the Span-KD (teacher-only d6) target build stage-by-stage for one utt.

For each inspected token u it prints:
  (1) occupancy gamma(t,u): which frames, from the DE-PEAKED posterior (WHERE)
  (2) at each occupancy frame: the RAW teacher posterior (WHAT source) top tokens
  (3) q_u = gamma-weighted avg of RAW posterior, BEFORE dropping blank
  (4) q_u AFTER drop-blank + renormalize  (== the stored top-k target)
so we can see exactly where the distribution collapses to ~one-hot.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


def blank_penalty_logprobs(log_probs, blank_id, delta):
    if delta == 0.0:
        return log_probs
    lp = log_probs.copy()
    lp[:, blank_id] = log_probs[:, blank_id] - delta
    m = np.max(lp, axis=-1, keepdims=True)
    lp = lp - (m + np.log(np.exp(lp - m).sum(axis=-1, keepdims=True)))
    return lp


def main():
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train_clean_100.teacher_only_d6.json")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--tokens", default="", help="comma token indices; default: first 2 + rarest")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(dev).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tk = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    name = lambda i: ("<blank>" if int(i) == blank else tk.ids_to_tokens([int(i)])[0])

    row = [json.loads(l) for l in open(args.manifest)][args.idx]
    bpe_ids = [int(i) for i in tk.text_to_ids(row["text"]) if int(i) != blank]
    wav, wsr = sf.read(row["audio_filepath"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    print(f"utt idx={args.idx}  text={row['text'][:80]!r}")
    print(f"n_tokens={len(bpe_ids)}  blank_id={blank}\n")

    with torch.no_grad():
        wt = torch.from_numpy(wav).float().unsqueeze(0).to(dev)
        wl = torch.tensor([wt.shape[1]], device=dev)
        lp, el, _ = teacher.forward(input_signal=wt, input_signal_length=wl)
        T = int(el[0])
        raw_lp = lp[0, :T].cpu().float().numpy()          # RAW (WHAT source)
    raw_p = np.exp(raw_lp)
    dep_lp = blank_penalty_logprobs(raw_lp, blank, args.delta)   # de-peaked (WHERE)
    gamma = ctc_forward_backward(dep_lp, bpe_ids, blank)["token_gamma"]  # (T,N)

    # global peakiness of the RAW posterior (context)
    nb = raw_p.copy(); nb[:, blank] = 0
    print(f"[raw posterior] mean p(blank)={raw_p[:, blank].mean():.3f}   "
          f"frames with p(blank)>0.9: {(raw_p[:, blank] > 0.9).mean():.1%}\n")

    # pick tokens: first 2 + the one whose FINAL target is least one-hot
    finals = []
    for u in range(len(bpe_ids)):
        q = (gamma[:, u:u+1] * raw_p).sum(0)
        q[blank] = 0.0
        q = q / max(q.sum(), 1e-12)
        finals.append(q)
    top1 = np.array([q.max() for q in finals])
    if args.tokens:
        sel = [int(x) for x in args.tokens.split(",")]
    else:
        sel = sorted(set([0, min(1, len(bpe_ids)-1), int(top1.argmin())]))

    for u in sel:
        tgt_id = bpe_ids[u]
        print("=" * 72)
        print(f"TOKEN u={u}  transcript id={tgt_id} ({name(tgt_id)!r})   "
              f"final-target top1={top1[u]:.3f}")
        occ = np.where(gamma[:, u] > 0.01)[0]
        print(f"\n(1) occupancy gamma(t,u) frames (de-peaked WHERE): {list(occ)}")
        print(f"    gamma values: {[f'{gamma[t,u]:.2f}' for t in occ]}")

        print(f"\n(2) RAW teacher posterior at each occupancy frame (WHAT source):")
        for t in occ:
            order = np.argsort(raw_p[t])[::-1][:4]
            parts = [f"{name(i)!r}={raw_p[t,i]:.3f}" for i in order]
            print(f"    t={t:3d} g={gamma[t,u]:.2f} | p(blank)={raw_p[t,blank]:.3f} | "
                  + "  ".join(parts))

        q = (gamma[:, u:u+1] * raw_p).sum(0)
        print(f"\n(3) q_u = gamma-weighted avg of RAW posterior (BEFORE drop-blank):")
        order = np.argsort(q)[::-1][:5]
        print(f"    p(blank)={q[blank]:.3f} | "
              + "  ".join(f"{name(i)!r}={q[i]:.3f}" for i in order if i != blank))

        q2 = q.copy(); q2[blank] = 0.0; q2 = q2 / max(q2.sum(), 1e-12)
        order2 = np.argsort(q2)[::-1][:8]
        print(f"\n(4) FINAL target = drop-blank + renorm, top-8 (stored):")
        for i in order2:
            print(f"    {name(i)!r:12s} {q2[i]:.4f}")
        print()


if __name__ == "__main__":
    main()
