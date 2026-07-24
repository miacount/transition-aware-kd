#!/usr/bin/env python3
"""Validate the blank-suppressed (de-peaked) occupancy against MFA forced alignments.

Tests whether delta widening is MEANINGFUL (H1: mass spreads into the token's true
acoustic interval, asymmetrically toward onset, width tracks duration) or just
BLANK-KILLING (H2: symmetric smear indistinguishable from a fixed kernel).

Metrics (per delta in --deltas, plus a width-matched symmetric-kernel control):
  A. coverage        : occupancy mass inside the token's MFA word interval (+/- tol)
                       and coverage of the ADDED mass (gamma_d - gamma_0)+
  B. asymmetry       : signed center shift of added mass vs the delta=0 center
                       (negative = earlier/toward onset); spike position within word
  C. width~duration  : effective width (participation ratio) vs MFA word duration
  D. residual identity at blank frames (the D3 precondition): after removing blank
     and renormalizing, does the remaining non-blank mass belong to the ADJACENT
     transcript token (or the MFA word covering the frame), or to unrelated tokens?
     Stratified by distance to nearest spike and by whether delta=6 recruited it.
  E. target purity   : share of the gamma-weighted WHAT target on y_u itself /
                       transcript neighbours / other transcript / out-of-transcript,
                       for the full target and for the added-mass part only.

Outputs: printed report + summary.json + per-token/per-frame npz records.
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

from visualize_ctc_forward_backward import ctc_forward_backward, make_ext_labels  # noqa: E402

NEG = -np.inf


# ---------------------------------------------------------------- FB (fast) --
def fb_token_gamma_fast(log_probs, tokens, blank_id):
    """Vectorized CTC forward-backward; same semantics as ctc_forward_backward."""
    ext = make_ext_labels(tokens, blank_id)
    T = log_probs.shape[0]
    S = len(ext)
    em = log_probs[:, ext].astype(np.float64)  # (T, S)

    allow2 = np.zeros(S, dtype=bool)
    if S > 2:
        allow2[2:] = (ext[2:] != blank_id) & (ext[2:] != ext[:-2])
    allow2_next = np.concatenate([allow2[2:], [False, False]])[:S]

    alpha = np.full((T, S), NEG)
    alpha[0, 0] = em[0, 0]
    if S > 1:
        alpha[0, 1] = em[0, 1]
    for t in range(1, T):
        prev = alpha[t - 1]
        m1 = np.concatenate(([NEG], prev[:-1]))
        m2 = np.concatenate(([NEG, NEG], prev[:-2])) if S > 2 else np.full(S, NEG)
        m2 = np.where(allow2, m2, NEG)
        alpha[t] = np.logaddexp(np.logaddexp(prev, m1), m2) + em[t]

    beta = np.full((T, S), NEG)
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        v = beta[t + 1] + em[t + 1]
        n1 = np.concatenate((v[1:], [NEG]))
        n2 = np.concatenate((v[2:], [NEG, NEG])) if S > 2 else np.full(S, NEG)
        n2 = np.where(allow2_next, n2, NEG)
        beta[t] = np.logaddexp(np.logaddexp(v, n1), n2)

    if S == 1:
        log_z = alpha[T - 1, 0]
    else:
        log_z = np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    gamma_state = np.exp(alpha + beta - log_z)
    gamma_state = np.nan_to_num(gamma_state, nan=0.0, posinf=0.0, neginf=0.0)
    return gamma_state[:, 1::2]  # (T, N)


def blank_penalty_logprobs(log_probs, blank_id, delta):
    """Same as build_span_kd_targets.blank_penalty_logprobs."""
    if delta == 0.0:
        return log_probs
    lp = log_probs.copy()
    lp[:, blank_id] = log_probs[:, blank_id] - delta
    m = np.max(lp, axis=-1, keepdims=True)
    return lp - (m + np.log(np.exp(lp - m).sum(axis=-1, keepdims=True)))


# ----------------------------------------------------------------- helpers --
def normalize_cols_occ(gamma):
    """(T, N) -> per-token occupancy distribution over time (each col sums to 1)."""
    s = gamma.sum(axis=0, keepdims=True)
    return gamma / np.maximum(s, 1e-12)


def pr_width(occ):
    """Participation-ratio effective width per token. occ: (T, N) col-normalized."""
    return 1.0 / np.maximum((occ ** 2).sum(axis=0), 1e-12)


def gauss_kernel(sigma, radius=None):
    if radius is None:
        radius = max(1, int(np.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def smooth_occ(occ, kernel):
    """Convolve each token's occupancy along time (edges truncated+renorm)."""
    T, N = occ.shape
    out = np.empty_like(occ)
    for u in range(N):
        out[:, u] = np.convolve(occ[:, u], kernel, mode="same")
    return normalize_cols_occ(out)


def group_bpe_words(tokens_str):
    """SentencePiece tokens -> list of (start_idx, end_idx) per word."""
    groups = []
    for i, tok in enumerate(tokens_str):
        if tok.startswith("▁") or i == 0:
            groups.append([i, i + 1])
        else:
            groups[-1][1] = i + 1
    return groups


def read_manifest(path, limit):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def run_teacher(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


def frame_window(start_s, end_s, fd, T, tol_frames):
    """Frame indices whose center lies in [start - tol*fd, end + tol*fd)."""
    lo = start_s - tol_frames * fd
    hi = end_s + tol_frames * fd
    t0 = max(0, int(np.floor(lo / fd - 0.5 + 1e-9)) )
    t1 = min(T, int(np.ceil(hi / fd - 0.5 - 1e-9)) + 1)
    idx = np.arange(t0, t1)
    centers = (idx + 0.5) * fd
    return idx[(centers >= lo) & (centers < hi)]


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--alignments", required=True, help="json: utt_id -> {words:[{word,start,end}],...}")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--deltas", default="0,3,6,9,12")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tol", type=int, default=1, help="coverage tolerance in frames")
    ap.add_argument("--recruit-eps", type=float, default=0.01,
                    help="occupancy gain threshold: frame counted as recruited by delta")
    ap.add_argument("--ctl-sigma", type=float, default=0.0,
                    help="if >0, build the symmetric-kernel control inline with this sigma "
                         "(frames) so it enters the D/E identity metrics too")
    ap.add_argument("--out", default="analysis/mfa_validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    assert deltas[0] == 0.0, "first delta must be 0 (reference)"
    ref_delta = 6.0 if 6.0 in deltas else deltas[1]

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
    utts = []          # per-utt dict of arrays for post-pass
    verified_fb = False

    # D-metric accumulators (per blank frame records)
    bf = {k: [] for k in ["dist", "pblank", "share_adj", "top1_adj", "share_tr",
                          "share_mfa", "top1_mfa", "in_mfa_word", "recruited",
                          "cross_word", "share_out", "top1_id", "adjL_id", "adjR_id"]}

    pblank_all = []

    # per-(frame, token) residual identity of RECRUITED mass, per widening variant
    rr_names = [f"delta={d:g}" for d in deltas[1:]] + (["ctl"] if args.ctl_sigma > 0 else [])
    rr = {v: {k: [] for k in ["r_yu", "w", "in_word", "blankdom"]} for v in rr_names}

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
        lp = run_teacher(teacher, wav, args.device)          # (T, V) sharp
        T = lp.shape[0]
        fd = (len(wav) / sr) / T                              # sec per frame
        probs = np.exp(lp)
        pb = probs[:, blank]
        pblank_all.append(pb)
        N = len(bpe_ids)

        # token -> word index, word -> (start,end) sec
        tok_word = np.zeros(N, dtype=np.int64)
        for wi, (a, b) in enumerate(groups):
            tok_word[a:b] = wi
        w_start = np.array([w["start"] for w in words_mfa])
        w_end = np.array([w["end"] for w in words_mfa])

        occ = {}
        for d in deltas:
            g = fb_token_gamma_fast(blank_penalty_logprobs(lp, blank, d), bpe_ids, blank)
            if not verified_fb:
                g_ref = ctc_forward_backward(
                    blank_penalty_logprobs(lp, blank, d), bpe_ids, blank)["token_gamma"]
                assert np.allclose(g, g_ref, atol=1e-8), "fast FB != reference FB"
                verified_fb = True
                print("[ok] fast FB verified against reference implementation")
            occ[d] = normalize_cols_occ(g)
        if args.ctl_sigma > 0:
            occ["ctl"] = smooth_occ(occ[0.0], gauss_kernel(args.ctl_sigma))

        spikes = occ[0.0].argmax(axis=0)                     # (N,) delta=0 spike frame

        # ------- recruited-(t,u) residual identity: delta widening vs kernel -
        resid = probs.copy()
        resid[:, blank] = 0.0
        resid = resid / np.maximum(resid.sum(axis=1, keepdims=True), 1e-12)
        blankdom = probs.argmax(axis=1) == blank
        ids_arr = np.array(bpe_ids)
        variants_rr = [(f"delta={d:g}", d) for d in deltas[1:]] + \
            ([("ctl", "ctl")] if args.ctl_sigma > 0 else [])
        for vname, vkey in variants_rr:
            A = np.maximum(occ[vkey] - occ[0.0], 0.0)
            ts_r, us_r = np.where(A > args.recruit_eps)
            if len(ts_r):
                wis = tok_word[us_r]
                centers = (ts_r + 0.5) * fd
                inw = ((w_start[wis] - fd) <= centers) & (centers < (w_end[wis] + fd))
                rr[vname]["r_yu"].extend(resid[ts_r, ids_arr[us_r]].tolist())
                rr[vname]["w"].extend(A[ts_r, us_r].tolist())
                rr[vname]["in_word"].extend(inw.astype(float).tolist())
                rr[vname]["blankdom"].extend(blankdom[ts_r].astype(float).tolist())

        # ---------------- D: blank-frame residual identity (sharp posterior) --
        add_ref = np.maximum(occ[ref_delta] - occ[0.0], 0.0)  # (T, N)
        blank_frames = np.where(probs.argmax(axis=1) == blank)[0]
        ids_arr = np.array(bpe_ids)
        tr_set = set(bpe_ids)
        for t in blank_frames:
            res = probs[t].copy()
            res[blank] = 0.0
            tot = res.sum()
            if tot < 1e-12:
                continue
            res /= tot
            left = np.where(spikes <= t)[0]
            right = np.where(spikes > t)[0]
            uL = left[np.argmax(spikes[left])] if len(left) else None
            uR = right[np.argmin(spikes[right])] if len(right) else None
            adj_ids = {int(ids_arr[u]) for u in (uL, uR) if u is not None}
            dcands = []
            if uL is not None:
                dcands.append(t - spikes[uL])
            if uR is not None:
                dcands.append(spikes[uR] - t)
            dist = int(min(dcands))
            top1 = int(res.argmax())
            share_tr = float(res[list(tr_set)].sum())
            share_adj = float(res[list(adj_ids)].sum())
            cross = (uL is not None and uR is not None
                     and tok_word[uL] != tok_word[uR])
            # MFA word covering this frame center
            c = (t + 0.5) * fd
            wi = np.where((w_start <= c) & (c < w_end))[0]
            if len(wi):
                w_tok_ids = {int(i) for i in ids_arr[groups[wi[0]][0]:groups[wi[0]][1]]}
                share_mfa = float(res[list(w_tok_ids)].sum())
                top1_mfa = float(top1 in w_tok_ids)
                in_word = 1.0
            else:
                share_mfa, top1_mfa, in_word = 0.0, 0.0, 0.0
            bf["dist"].append(dist)
            bf["pblank"].append(float(pb[t]))
            bf["share_adj"].append(share_adj)
            bf["top1_adj"].append(float(top1 in adj_ids))
            bf["share_tr"].append(share_tr)
            bf["share_out"].append(1.0 - share_tr)
            bf["share_mfa"].append(share_mfa)
            bf["top1_mfa"].append(top1_mfa)
            bf["in_mfa_word"].append(in_word)
            bf["recruited"].append(float(add_ref[t].max() > args.recruit_eps))
            bf["cross_word"].append(float(cross))
            bf["top1_id"].append(top1)
            bf["adjL_id"].append(int(ids_arr[uL]) if uL is not None else -1)
            bf["adjR_id"].append(int(ids_arr[uR]) if uR is not None else -1)

        # ------------- E: WHAT-target composition per delta (full & added) ---
        nb = probs.copy()
        nb[:, blank] = 0.0                                   # non-blank posterior
        tgt = {}
        evariants = deltas + (["ctl"] if args.ctl_sigma > 0 else [])
        for d in evariants:
            for part, wmat in (("full", occ[d]), ("added", np.maximum(occ[d] - occ[0.0], 0.0))):
                q = wmat.T @ nb                              # (N, V) unnormalized
                qs = q.sum(axis=1)
                valid = qs > 1e-9
                q = q / np.maximum(qs[:, None], 1e-12)
                self_share = q[np.arange(N), ids_arr]
                nb_ids_prev = np.concatenate([[- 1], ids_arr[:-1]])
                nb_ids_next = np.concatenate([ids_arr[1:], [-1]])
                nbr_share = np.zeros(N)
                for u in range(N):
                    for v in (nb_ids_prev[u], nb_ids_next[u]):
                        if v >= 0 and v != ids_arr[u]:
                            nbr_share[u] += q[u, v]
                tr_share = q[:, sorted(tr_set)].sum(axis=1)
                other_tr = tr_share - self_share - nbr_share
                nb_tot = nb.sum(axis=1)                      # (T,) = 1 - p_blank
                eff = (wmat.T @ nb_tot) / np.maximum(occ[d].T @ nb_tot, 1e-12)
                tgt[(d, part)] = dict(self=self_share, nbr=nbr_share,
                                      other_tr=np.maximum(other_tr, 0.0),
                                      out=1.0 - tr_share, valid=valid,
                                      mass=wmat.sum(axis=0), eff_share=eff)

        utts.append(dict(utt=utt_id, T=T, fd=fd, ids=ids_arr, tok_word=tok_word,
                         groups=groups, w_start=w_start, w_end=w_end,
                         occ=occ, spikes=spikes, tgt=tgt))

    print(f"\nutts kept={len(utts)} skipped={skipped}")
    pb_cat = np.concatenate(pblank_all)
    print(f"sanity: mean p(blank)={pb_cat.mean():.3f} (report says ~0.816)", flush=True)

    # ---------------- control: symmetric kernel matched to ref_delta width ---
    def all_widths(get_occ):
        out = []
        for U in utts:
            out.append(pr_width(get_occ(U)))
        return np.concatenate(out)

    w0 = all_widths(lambda U: U["occ"][0.0])
    wref = all_widths(lambda U: U["occ"][ref_delta])
    target_w = wref.mean()

    if args.ctl_sigma > 0:
        sigma = args.ctl_sigma
    else:
        lo_s, hi_s = 0.05, 6.0
        for _ in range(40):
            mid = 0.5 * (lo_s + hi_s)
            k = gauss_kernel(mid)
            wm = np.concatenate([pr_width(smooth_occ(U["occ"][0.0], k)) for U in utts]).mean()
            if wm < target_w:
                lo_s = mid
            else:
                hi_s = mid
        sigma = 0.5 * (lo_s + hi_s)
        kernel = gauss_kernel(sigma)
        for U in utts:
            U["occ"]["ctl"] = smooth_occ(U["occ"][0.0], kernel)
    print(f"[control] gaussian sigma={sigma:.3f} frames "
          f"(mean width {np.concatenate([pr_width(U['occ']['ctl']) for U in utts]).mean():.3f} "
          f"vs delta={ref_delta} width {target_w:.3f}, delta=0 width {w0.mean():.3f})")

    # ---------------- A/B/C per variant ---------------------------------------
    variants = deltas + ["ctl"]
    recs = {v: {k: [] for k in ["cov0", "cov1", "addcov1", "shift_f", "width",
                                "dur_f", "single", "spike_pos"]} for v in variants}
    for U in utts:
        T, fd = U["T"], U["fd"]
        o0 = U["occ"][0.0]
        N = len(U["ids"])
        for v in variants:
            o = U["occ"][v]
            add = np.maximum(o - o0, 0.0)
            for u in range(N):
                wi = U["tok_word"][u]
                s_s, e_s = U["w_start"][wi], U["w_end"][wi]
                if e_s - s_s <= 0:
                    continue
                win0 = frame_window(s_s, e_s, fd, T, 0)
                win1 = frame_window(s_s, e_s, fd, T, args.tol)
                cov0 = float(o[win0, u].sum()) if len(win0) else 0.0
                cov1 = float(o[win1, u].sum()) if len(win1) else 0.0
                asum = add[:, u].sum()
                addcov1 = float(add[win1, u].sum() / asum) if asum > 1e-6 else np.nan
                c0 = float((np.arange(T) * o0[:, u]).sum())
                shift = (float((np.arange(T) * add[:, u]).sum() / asum) - c0) if asum > 1e-6 else np.nan
                a, b = U["groups"][wi]
                single = (b - a) == 1
                spike_pos = ((U["spikes"][u] + 0.5) * fd - s_s) / (e_s - s_s)
                recs[v]["cov0"].append(cov0)
                recs[v]["cov1"].append(cov1)
                recs[v]["addcov1"].append(addcov1)
                recs[v]["shift_f"].append(shift)
                recs[v]["width"].append(float(pr_width(o[:, u:u + 1])[0]))
                recs[v]["dur_f"].append((e_s - s_s) / fd)
                recs[v]["single"].append(single)
                recs[v]["spike_pos"].append(float(spike_pos))

    # ---------------- report ---------------------------------------------------
    from scipy import stats as sps

    def name(v):
        return f"delta={v:g}" if isinstance(v, float) else "kernel-ctl"

    summary = {"n_utts": len(utts), "skipped": skipped, "sigma_ctl": sigma,
               "mean_pblank": float(pb_cat.mean()), "variants": {}}

    print("\n================= A/B/C: occupancy vs MFA word intervals =================")
    hdr = (f"{'variant':>12} {'cov(strict)':>11} {'cov(+/-1)':>10} {'addcov(+/-1)':>12} "
           f"{'shift(frames)':>13} {'width':>7} {'r_width~dur(single)':>20}")
    print(hdr)
    for v in variants:
        r = recs[v]
        single = np.array(r["single"])
        w = np.array(r["width"])
        d = np.array(r["dur_f"])
        sel = single & np.isfinite(w) & np.isfinite(d)
        if sel.sum() > 10 and np.std(w[sel]) > 1e-9:
            pear = float(sps.pearsonr(w[sel], d[sel])[0])
            spear = float(sps.spearmanr(w[sel], d[sel])[0])
        else:
            pear = spear = float("nan")
        addcov = np.array(r["addcov1"], dtype=np.float64)
        shift = np.array(r["shift_f"], dtype=np.float64)
        row = dict(cov_strict=float(np.mean(r["cov0"])), cov_tol=float(np.mean(r["cov1"])),
                   addcov_tol=float(np.nanmean(addcov)) if np.isfinite(addcov).any() else None,
                   shift_frames=float(np.nanmean(shift)) if np.isfinite(shift).any() else None,
                   shift_med=float(np.nanmedian(shift)) if np.isfinite(shift).any() else None,
                   width=float(np.mean(w)), pearson_width_dur=pear, spearman_width_dur=spear,
                   n_tokens=len(r["cov0"]))
        summary["variants"][name(v)] = row
        print(f"{name(v):>12} {row['cov_strict']:>11.3f} {row['cov_tol']:>10.3f} "
              f"{(row['addcov_tol'] if row['addcov_tol'] is not None else float('nan')):>12.3f} "
              f"{(row['shift_frames'] if row['shift_frames'] is not None else float('nan')):>13.3f} "
              f"{row['width']:>7.2f} {pear:>9.3f}/{spear:.3f}")

    sp = np.array(recs[deltas[0]]["spike_pos"], dtype=np.float64)
    sp = sp[np.isfinite(sp)]
    summary["spike_pos_in_word"] = dict(mean=float(sp.mean()), med=float(np.median(sp)),
                                        frac_after_mid=float((sp > 0.5).mean()),
                                        frac_outside=float(((sp < 0) | (sp > 1)).mean()))
    print(f"\nspike position within MFA word [0=onset,1=offset]: mean={sp.mean():.3f} "
          f"median={np.median(sp):.3f}  P(pos>0.5)={(sp > 0.5).mean():.3f} "
          f"P(outside word)={((sp < 0) | (sp > 1)).mean():.3f}")

    print("\n================= D: blank-frame residual identity =================")
    B = {k: np.array(v, dtype=np.float64) for k, v in bf.items()}
    ntot = len(B["dist"])
    print(f"blank frames analyzed: {ntot}  (mean p_blank={B['pblank'].mean():.3f})")
    summary["blank_frames"] = {"n": ntot}

    def dstat(mask, label):
        n = int(mask.sum())
        if n == 0:
            return None
        row = dict(n=n,
                   share_adj=float(B["share_adj"][mask].mean()),
                   top1_adj=float(B["top1_adj"][mask].mean()),
                   share_transcript=float(B["share_tr"][mask].mean()),
                   share_out=float(B["share_out"][mask].mean()),
                   share_mfa_word=float(B["share_mfa"][mask].mean()),
                   top1_mfa_word=float(B["top1_mfa"][mask].mean()))
        print(f"{label:>28}: n={n:>6}  adj_share={row['share_adj']:.3f} "
              f"adj_top1={row['top1_adj']:.3f}  transcript={row['share_transcript']:.3f} "
              f"out={row['share_out']:.3f}  mfa_word_share={row['share_mfa_word']:.3f} "
              f"mfa_top1={row['top1_mfa_word']:.3f}")
        return row

    summary["blank_frames"]["all"] = dstat(np.ones(ntot, dtype=bool), "all blank frames")
    for dlo, dhi, lab in [(1, 1, "dist=1"), (2, 2, "dist=2"), (3, 3, "dist=3"), (4, 10 ** 9, "dist>=4")]:
        summary["blank_frames"][lab] = dstat((B["dist"] >= dlo) & (B["dist"] <= dhi), lab)
    summary["blank_frames"]["recruited_d6"] = dstat(B["recruited"] > 0.5,
                                                    f"recruited by delta={ref_delta:g}")
    summary["blank_frames"]["not_recruited"] = dstat(B["recruited"] < 0.5, "not recruited")
    summary["blank_frames"]["cross_word"] = dstat(B["cross_word"] > 0.5, "between words")
    summary["blank_frames"]["within_word"] = dstat(B["cross_word"] < 0.5, "within word")
    summary["blank_frames"]["in_mfa_word"] = dstat(B["in_mfa_word"] > 0.5, "inside an MFA word")

    print("\n----- recruited-(frame,token) residual identity: delta vs symmetric kernel -----")
    print("share of token u's OWN id in the frame's blank-removed residual, on frames the")
    print("variant recruited for u (occupancy gain > eps); weighted = by recruited mass")
    summary["recruited_identity"] = {}
    for vname in rr_names:
        lab = vname if vname != "ctl" else f"kernel-ctl(s={args.ctl_sigma:g})"
        R = {k: np.array(v, dtype=np.float64) for k, v in rr[vname].items()}
        if R["r_yu"].size == 0:
            continue
        for sub, m in (("all", np.ones(R["r_yu"].size, bool)), ("blank-dominated", R["blankdom"] > 0.5)):
            if m.sum() == 0:
                continue
            wgt = R["w"][m]
            row = dict(n=int(m.sum()),
                       r_yu_mean=float(R["r_yu"][m].mean()),
                       r_yu_weighted=float((R["r_yu"][m] * wgt).sum() / wgt.sum()),
                       in_word=float((R["in_word"][m] * wgt).sum() / wgt.sum()),
                       mass_total=float(wgt.sum()))
            summary["recruited_identity"][f"{lab}/{sub}"] = row
            print(f"{lab:>22} [{sub:>15}]: n={row['n']:>7}  r_yu={row['r_yu_mean']:.3f} "
                  f"(mass-wtd {row['r_yu_weighted']:.3f})  in-MFA-word(wtd)={row['in_word']:.3f}")

    print("\n================= E: WHAT-target composition per delta =================")
    print(f"{'variant':>12} {'part':>6} {'self(y_u)':>10} {'neighbour':>10} "
          f"{'other-transcript':>17} {'out-of-transcript':>18} {'occ-mass':>9} {'eff-wt':>7}")
    evariants_rep = deltas + (["ctl"] if args.ctl_sigma > 0 else [])
    for d in evariants_rep:
        for part in ("full", "added"):
            accs = {k: [] for k in ("self", "nbr", "other_tr", "out", "mass", "eff_share")}
            for U in utts:
                td = U["tgt"][(d, part)]
                m = td["valid"]
                if m.sum() == 0:
                    continue
                for k in accs:
                    accs[k].append(td[k][m])
            if not accs["self"]:
                continue
            row = {k: float(np.concatenate(v).mean()) for k, v in accs.items()}
            summary["variants"].setdefault(name(d), {})[f"target_{part}"] = row
            print(f"{name(d):>12} {part:>6} {row['self']:>10.3f} {row['nbr']:>10.3f} "
                  f"{row['other_tr']:>17.3f} {row['out']:>18.3f} "
                  f"{row['mass']:>9.3f} {row['eff_share']:>7.4f}")

    # ---------------- save -----------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out / "blank_frame_records.npz", **B)
    tok_np = {}
    for v in variants:
        key = name(v).replace("=", "").replace(".", "p")
        for k in ("cov0", "cov1", "addcov1", "shift_f", "width", "dur_f", "single"):
            tok_np[f"{key}_{k}"] = np.array(recs[v][k], dtype=np.float64)
    np.savez_compressed(out / "token_records.npz", **tok_np)
    print(f"\nsaved: {out}/summary.json, blank_frame_records.npz, token_records.npz")


if __name__ == "__main__":
    main()
