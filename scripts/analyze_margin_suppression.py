#!/usr/bin/env python3
"""Step-1 diagnostic for adaptive (margin-based) blank suppression.

Answers two questions BEFORE any training run:

  Q1 (separability): do MFA-silence frames and in-word shoulder frames occupy
     different blank-margin ranges? Adaptive suppression treats equal-margin
     frames identically, so if the two histograms overlap heavily no margin
     rule can protect silence while exposing shoulders — kill criterion.

  Q2 (outcome): per suppression variant — fixed delta vs adaptive
     delta_t = clip(m_t - mu, 0, dmax) — measure on the FB occupancy:
       silence leakage (occupancy mass on MFA-silence frames),
       width (participation ratio), word coverage (+/-1 tol),
       WHAT-target purity (self / neighbour / out-of-transcript shares).

Frame classes (from MFA word intervals + raw-teacher blank margin m_t):
  silence  : frame center outside every word interval +/- 1 frame
  edge     : within +/- 1 frame of a word boundary
  shoulder : strictly in-word, blank-dominant (m_t > 0)
  spike    : strictly in-word, non-blank-dominant (m_t <= 0)

Outputs: printed report + summary.json + margin_records.npz
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze_mfa_occupancy import (  # noqa: E402
    fb_token_gamma_fast, blank_penalty_logprobs, normalize_cols_occ,
    pr_width, group_bpe_words, read_manifest, run_teacher, frame_window,
)

NEG = -np.inf


def blank_margin(lp, blank_id, restrict_ids=None):
    """m_t = log p(blank|t) - max_v log p(v|t). restrict_ids: margin vs the
    transcript token set only (matches what the GT-constrained FB graph sees)."""
    nb = lp.copy()
    nb[:, blank_id] = NEG
    if restrict_ids is not None:
        mask = np.ones(lp.shape[1], dtype=bool)
        mask[list(restrict_ids)] = False
        mask[blank_id] = True  # already NEG
        nb[:, mask] = NEG
    return lp[:, blank_id] - nb.max(axis=1)


def margin_suppress_logprobs(lp, blank_id, mu, dmax, restrict_ids=None):
    """Adaptive suppression: delta_t = clip(m_t - mu, 0, dmax) on blank only,
    then per-frame renormalize (same convention as blank_penalty_logprobs)."""
    m = blank_margin(lp, blank_id, restrict_ids)
    delta_t = np.clip(m - mu, 0.0, dmax)
    out = lp.copy()
    out[:, blank_id] = lp[:, blank_id] - delta_t
    mx = np.max(out, axis=-1, keepdims=True)
    return out - (mx + np.log(np.exp(out - mx).sum(axis=-1, keepdims=True)))


def quantiles(x):
    if len(x) == 0:
        return None
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    return dict(n=int(len(x)), p5=float(q[0]), p25=float(q[1]), p50=float(q[2]),
                p75=float(q[3]), p95=float(q[4]), mean=float(np.mean(x)))


def overlap_coefficient(a, b, lo=-15.0, hi=25.0, bins=200):
    """Histogram overlap (0=disjoint, 1=identical) of two margin samples."""
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    w = (hi - lo) / bins
    return float(np.minimum(ha, hb).sum() * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--alignments", default="data/mfa/test_clean_alignments.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--deltas", default="0,3,6,9,12", help="fixed-delta variants")
    ap.add_argument("--mus", default="0,1,2", help="adaptive: allowed blank margin (nats)")
    ap.add_argument("--dmaxes", default="6,9,12", help="adaptive: suppression cap (nats)")
    ap.add_argument("--gt-margin", action="store_true",
                    help="add adaptive variants whose margin is vs transcript tokens only")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tol", type=int, default=1, help="word-interval tolerance in frames")
    ap.add_argument("--pause-frames", type=int, default=2,
                    help="min gap (frames) between words to count as a pause")
    ap.add_argument("--out", default="analysis/margin_suppression")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    mus = [float(m) for m in args.mus.split(",")]
    dmaxes = [float(d) for d in args.dmaxes.split(",")]

    variants = [("fixed", d, None) for d in deltas]
    variants += [("adapt", (mu, dm), None) for mu in mus for dm in dmaxes]
    if args.gt_margin:
        variants += [("adaptGT", (mu, dm), "gt") for mu in mus for dm in dmaxes]

    def vname(kind, p):
        if kind == "fixed":
            return f"delta={p:g}"
        tag = "muGT" if kind == "adaptGT" else "mu"
        return f"{tag}={p[0]:g},dmax={p[1]:g}"

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest, args.limit)
    with open(args.alignments) as f:
        mfa = json.load(f)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    skipped = {"no_mfa": 0, "word_mismatch": 0, "bpe_group": 0, "empty": 0}
    CLASSES = ("silence", "edge", "shoulder", "spike")
    margins = {c: [] for c in CLASSES}          # raw-margin samples per class
    margins_gt = {c: [] for c in CLASSES}       # transcript-restricted margin
    # token-level records per variant
    rec = {vname(k, p): {m: [] for m in ("width", "cov1", "sil", "pause_adj_sil",
                                         "self", "nbr", "out")}
           for k, p, _ in variants}
    n_utts = 0
    n_pause_adj_tokens = 0

    for row in rows:
        utt_id = Path(row["audio_filepath"]).stem
        if utt_id not in mfa:
            skipped["no_mfa"] += 1
            continue
        words_mfa = mfa[utt_id]["words"]
        words_txt = row["text"].split()
        if len(words_mfa) != len(words_txt) or any(
                w["word"].lower() != t for w, t in zip(words_mfa, words_txt)):
            skipped["word_mismatch"] += 1
            continue
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != blank]
        if not bpe_ids:
            skipped["empty"] += 1
            continue
        toks = tokenizer.ids_to_tokens(bpe_ids)
        groups = group_bpe_words(toks)
        if len(groups) != len(words_txt):
            skipped["bpe_group"] += 1
            continue

        wav, file_sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert file_sr == sr
        lp = run_teacher(teacher, wav, args.device)           # (T, V)
        T = lp.shape[0]
        fd = (len(wav) / sr) / T
        N = len(bpe_ids)
        ids_arr = np.array(bpe_ids)
        tr_set = sorted(set(bpe_ids))

        w_start = np.array([w["start"] for w in words_mfa])
        w_end = np.array([w["end"] for w in words_mfa])
        tok_word = np.zeros(N, dtype=np.int64)
        for wi, (a, b) in enumerate(groups):
            tok_word[a:b] = wi

        # ---------------- frame classes -----------------------------------
        centers = (np.arange(T) + 0.5) * fd
        in_word0 = np.zeros(T, dtype=bool)   # strictly inside a word
        in_word1 = np.zeros(T, dtype=bool)   # inside +/- tol frames
        for s, e in zip(w_start, w_end):
            in_word0 |= (centers >= s) & (centers < e)
            in_word1 |= (centers >= s - args.tol * fd) & (centers < e + args.tol * fd)
        m_raw = blank_margin(lp, blank)
        m_gt = blank_margin(lp, blank, restrict_ids=tr_set)
        cls = np.full(T, "spike", dtype=object)
        cls[~in_word1] = "silence"
        cls[in_word1 & ~in_word0] = "edge"
        cls[in_word0 & (m_raw > 0)] = "shoulder"
        for c in CLASSES:
            sel = cls == c
            margins[c].append(m_raw[sel])
            margins_gt[c].append(m_gt[sel])
        sil_frames = ~in_word1

        # pause-adjacent tokens: word borders a >= pause_frames inter-word gap
        # (or utterance-boundary silence of that size)
        gap_before = np.concatenate([[w_start[0]], w_start[1:] - w_end[:-1]])
        gap_after = np.concatenate([w_start[1:] - w_end[:-1],
                                    [len(wav) / sr - w_end[-1]]])
        pause_thr = args.pause_frames * fd
        word_pause_adj = (gap_before >= pause_thr) | (gap_after >= pause_thr)
        tok_pause_adj = word_pause_adj[tok_word]
        n_pause_adj_tokens += int(tok_pause_adj.sum())

        # non-blank posterior for WHAT-target purity
        probs = np.exp(lp)
        nbp = probs.copy()
        nbp[:, blank] = 0.0
        prev_ids = np.concatenate([[-1], ids_arr[:-1]])
        next_ids = np.concatenate([ids_arr[1:], [-1]])

        # ---------------- per-variant FB + metrics ------------------------
        for kind, p, _ in variants:
            if kind == "fixed":
                lp_s = blank_penalty_logprobs(lp, blank, p)
            elif kind == "adapt":
                lp_s = margin_suppress_logprobs(lp, blank, p[0], p[1])
            else:  # adaptGT
                lp_s = margin_suppress_logprobs(lp, blank, p[0], p[1], restrict_ids=tr_set)
            occ = normalize_cols_occ(fb_token_gamma_fast(lp_s, bpe_ids, blank))
            r = rec[vname(kind, p)]
            r["width"].extend(pr_width(occ).tolist())
            sil_mass = occ[sil_frames].sum(axis=0)            # (N,)
            r["sil"].extend(sil_mass.tolist())
            r["pause_adj_sil"].extend(sil_mass[tok_pause_adj].tolist())
            for u in range(N):
                wi = tok_word[u]
                win1 = frame_window(w_start[wi], w_end[wi], fd, T, args.tol)
                r["cov1"].append(float(occ[win1, u].sum()) if len(win1) else 0.0)
            q = occ.T @ nbp                                    # (N, V)
            q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
            self_share = q[np.arange(N), ids_arr]
            nbr_share = np.zeros(N)
            for u in range(N):
                for v in (prev_ids[u], next_ids[u]):
                    if v >= 0 and v != ids_arr[u]:
                        nbr_share[u] += q[u, v]
            tr_share = q[:, tr_set].sum(axis=1)
            r["self"].extend(self_share.tolist())
            r["nbr"].extend(nbr_share.tolist())
            r["out"].extend((1.0 - tr_share).tolist())

        n_utts += 1

    print(f"\nutts kept={n_utts} skipped={skipped}")
    margins = {c: np.concatenate(v) if v else np.array([]) for c, v in margins.items()}
    margins_gt = {c: np.concatenate(v) if v else np.array([]) for c, v in margins_gt.items()}
    total_frames = sum(len(v) for v in margins.values())
    summary = {"n_utts": n_utts, "skipped": skipped, "n_frames": int(total_frames),
               "classes": {}, "variants": {}}

    # ================= Q1: margin distribution by frame class ==============
    print("\n========== Q1: raw blank margin m_t by MFA frame class ==========")
    print(f"{'class':>10} {'frac':>6} {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'mean':>7}")
    for c in CLASSES:
        st = quantiles(margins[c])
        if st is None:
            continue
        st["frac"] = st["n"] / max(total_frames, 1)
        st["gt"] = quantiles(margins_gt[c])
        summary["classes"][c] = st
        print(f"{c:>10} {st['frac']:>6.3f} {st['p5']:>7.2f} {st['p25']:>7.2f} "
              f"{st['p50']:>7.2f} {st['p75']:>7.2f} {st['p95']:>7.2f} {st['mean']:>7.2f}")

    ov = overlap_coefficient(margins["silence"], margins["shoulder"])
    ov_gt = overlap_coefficient(margins_gt["silence"], margins_gt["shoulder"])
    summary["overlap_sil_shoulder"] = ov
    summary["overlap_sil_shoulder_gt"] = ov_gt
    print(f"\nsilence-vs-shoulder margin overlap coefficient: raw={ov:.3f}  "
          f"GT-restricted={ov_gt:.3f}   (0=separable, 1=identical)")
    # decision-relevant threshold sweep: mass of each class above candidate mu
    print(f"\n{'mu':>5} {'P(sil m>mu)':>12} {'P(shld m>mu)':>13}   "
          "(frames with m>mu are the ones adaptive suppression touches)")
    summary["mu_sweep"] = {}
    for mu in [0, 1, 2, 3, 4, 6, 8]:
        ps = float((margins["silence"] > mu).mean()) if len(margins["silence"]) else float("nan")
        ph = float((margins["shoulder"] > mu).mean()) if len(margins["shoulder"]) else float("nan")
        summary["mu_sweep"][str(mu)] = dict(sil=ps, shoulder=ph)
        print(f"{mu:>5} {ps:>12.3f} {ph:>13.3f}")

    # ================= Q2: FB outcome per suppression variant ==============
    print("\n========== Q2: occupancy outcome per suppression variant ==========")
    print(f"{'variant':>18} {'width':>6} {'cov(+/-1)':>9} {'sil-leak':>9} {'sil-p90':>8} "
          f"{'pauseadj':>9} {'self':>6} {'nbr':>6} {'out':>6}")
    for kind, p, _ in variants:
        nm = vname(kind, p)
        r = rec[nm]
        sil = np.array(r["sil"])
        pas = np.array(r["pause_adj_sil"])
        row = dict(width=float(np.mean(r["width"])),
                   cov1=float(np.mean(r["cov1"])),
                   sil_leak=float(sil.mean()),
                   sil_p90=float(np.quantile(sil, 0.9)),
                   pause_adj_sil=float(pas.mean()) if len(pas) else float("nan"),
                   self_share=float(np.mean(r["self"])),
                   nbr_share=float(np.mean(r["nbr"])),
                   out_share=float(np.mean(r["out"])),
                   n_tokens=len(sil))
        summary["variants"][nm] = row
        print(f"{nm:>18} {row['width']:>6.2f} {row['cov1']:>9.3f} {row['sil_leak']:>9.4f} "
              f"{row['sil_p90']:>8.4f} {row['pause_adj_sil']:>9.4f} "
              f"{row['self_share']:>6.3f} {row['nbr_share']:>6.3f} {row['out_share']:>6.3f}")
    print(f"\n(sil-leak = mean occupancy mass on MFA-silence frames per token; "
          f"pauseadj = same over the {n_pause_adj_tokens} tokens bordering a "
          f">={args.pause_frames}-frame pause; self/nbr/out = WHAT-target shares)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out / "margin_records.npz",
                        **{f"m_{c}": margins[c] for c in CLASSES},
                        **{f"mgt_{c}": margins_gt[c] for c in CLASSES})
    print(f"\nsaved: {out}/summary.json, margin_records.npz")


if __name__ == "__main__":
    main()
