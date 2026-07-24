#!/usr/bin/env python3
"""Spike-aligned profile of the ATD support, and what is acoustically there.

Over test-clean, aligned on each token's delta=0 spike:

  - mean teacher occupancy at each offset, per delta (+ a width-matched
    symmetric Gaussian control)
  - the acoustic ceiling: at each offset, how much of the blank-removed residual
    posterior is the token's OWN id, and whether the frame is still inside the
    token's MFA word.

The point: token identity survives only at +/-1 frame, which is exactly the span
delta=6 recruits. Backs the numbers in analysis/VALIDATION_FIGURES.md (Q1).
Raw output: analysis/mfa_validation/profile.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))

from analyze_mfa_occupancy import (  # noqa: E402
    blank_penalty_logprobs,
    fb_token_gamma_fast,
    gauss_kernel,
    group_bpe_words,
    normalize_cols_occ,
    pr_width,
    read_manifest,
    run_teacher,
    smooth_occ,
)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--alignments", default="data/mfa/test_clean_alignments.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--deltas", default="0,3,6,9,12")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--radius", type=int, default=6, help="profile window, frames")
    ap.add_argument("--recruit-eps", type=float, default=0.01)
    ap.add_argument("--summary", default="analysis/mfa_validation/summary.json",
                    help="reuse composition numbers from analyze_mfa_occupancy.py")
    ap.add_argument("--out-json", default="analysis/mfa_validation/profile.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    R = args.radius
    rel = np.arange(-R, R + 1)

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    with open(args.alignments) as f:
        mfa = json.load(f)
    rows = read_manifest(args.manifest, args.limit)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    prof = {d: [] for d in deltas}          # spike-aligned occupancy profiles
    prof_ctl = []
    resid_id, in_word = [], []              # per-token, per-offset
    n_tok = 0

    # pass 1: teacher posteriors + occupancies (the control sigma needs them all first)
    cache = []
    for row in rows:
        utt = Path(row["audio_filepath"]).stem
        if utt not in mfa:
            continue
        words_mfa, words_txt = mfa[utt]["words"], row["text"].split()
        if len(words_mfa) != len(words_txt) or any(
                w["word"].lower() != t for w, t in zip(words_mfa, words_txt)):
            continue
        ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
        if not ids:
            continue
        toks = tokenizer.ids_to_tokens(ids)
        groups = group_bpe_words(toks)
        if len(groups) != len(words_txt):
            continue

        wav, file_sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        lp = run_teacher(teacher, wav, args.device)
        fd = (len(wav) / sr) / lp.shape[0]
        occ = {d: normalize_cols_occ(fb_token_gamma_fast(
            blank_penalty_logprobs(lp, blank, d), ids, blank)) for d in deltas}
        cache.append((lp.astype(np.float32), fd, ids, groups, words_mfa, occ))

    # width-matched symmetric-kernel control
    ref = 6.0 if 6.0 in deltas else deltas[1]
    target = np.concatenate([pr_width(c[5][ref]) for c in cache]).mean()
    lo_s, hi_s = 0.05, 6.0
    for _ in range(40):
        mid = 0.5 * (lo_s + hi_s)
        k = gauss_kernel(mid)
        wm = np.concatenate([pr_width(smooth_occ(c[5][0.0], k)) for c in cache]).mean()
        if wm < target:
            lo_s = mid
        else:
            hi_s = mid
    sigma = 0.5 * (lo_s + hi_s)
    print(f"[control] sigma={sigma:.3f} frames (matches delta={ref:g} width {target:.3f})")

    kernel = gauss_kernel(sigma)
    for lp, fd, ids, groups, words_mfa, occ in cache:
        T = lp.shape[0]
        N = len(ids)
        ids_arr = np.array(ids)
        probs = np.exp(lp)
        occ_ctl = smooth_occ(occ[0.0], kernel)

        res = probs.copy()
        res[:, blank] = 0.0
        res = res / np.maximum(res.sum(axis=1, keepdims=True), 1e-12)

        tok_word = np.zeros(N, dtype=np.int64)
        for wi, (a, b) in enumerate(groups):
            tok_word[a:b] = wi
        w_start = np.array([w["start"] for w in words_mfa])
        w_end = np.array([w["end"] for w in words_mfa])

        spikes = occ[0.0].argmax(axis=0)
        idx = spikes[:, None] + rel[None, :]                     # (N, 2R+1)
        ok = (idx >= 0) & (idx < T)
        idxc = np.clip(idx, 0, T - 1)

        for d in deltas:
            p = np.where(ok, occ[d][idxc, np.arange(N)[:, None]], np.nan)
            prof[d].append(p)
        prof_ctl.append(np.where(ok, occ_ctl[idxc, np.arange(N)[:, None]], np.nan))

        resid_id.append(np.where(ok, res[idxc, ids_arr[:, None]], np.nan))
        centers = (idxc + 0.5) * fd
        inw = (centers >= (w_start[tok_word][:, None] - fd)) & \
              (centers < (w_end[tok_word][:, None] + fd))
        in_word.append(np.where(ok, inw.astype(float), np.nan))
        n_tok += N

    P = {d: np.nanmean(np.concatenate(prof[d], 0), 0) for d in deltas}
    P_ctl = np.nanmean(np.concatenate(prof_ctl, 0), 0)
    RID = np.nanmean(np.concatenate(resid_id, 0), 0)
    INW = np.nanmean(np.concatenate(in_word, 0), 0)
    print(f"tokens={n_tok}  utts={len(cache)}")

    out = {"rel": rel.tolist(), "sigma_ctl": sigma, "n_tokens": int(n_tok),
           "profile": {f"delta={d:g}": P[d].tolist() for d in deltas},
           "profile_kernel_ctl": P_ctl.tolist(),
           "residual_self_id": RID.tolist(), "in_mfa_word": INW.tolist()}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"saved: {args.out_json}")
    print("\noffset : mean occupancy(delta=6) | residual self-id r(y_u) | inside MFA word")
    d6 = P[6.0] if 6.0 in P else P[deltas[1]]
    for j, r in enumerate(rel):
        if abs(r) <= 3:
            print(f"{r:>+6} : {d6[j]:>22.4f} | {RID[j]:>22.3f} | {INW[j]:>15.3f}")


if __name__ == "__main__":
    main()
