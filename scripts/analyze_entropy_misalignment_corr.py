#!/usr/bin/env python3
"""Method-4 pre-check: does a purely TEACHER-SIDE signal predict where the
student misaligns?

Hypothesis (adaptive-width span-KD): regions where the teacher is uncertain /
temporally smeared are exactly where the student's CTC spike lands off the
teacher's — so those regions need a WIDER span (more alignment tolerance).
If a teacher-only signal (computable offline at target-build time) correlates
with per-token student peak offset, an offline entropy-driven adaptive width is
justified and NON-CIRCULAR.

For each non-blank teacher segment k (from the token_avg manifest):
  target (misalignment):
    offset      = |s_T - t_S|   student peak vs teacher peak (student frames)
    wrong@t_T   = student argmax != token at the mapped teacher peak
  teacher-side signals (all offline, teacher-only):
    seg_len     = teacher segment length (frames)          [trivial baseline]
    H_peak      = full-vocab entropy at teacher peak frame
    H_span      = mean full-vocab entropy over span frames
    H_agg       = entropy of stored span-aggregated top-k target (avg_probs)
    unsharp     = 1 - teacher max prob at peak frame
    blank_frac  = fraction of span frames whose argmax is blank
    tspread     = temporal std of teacher token-mass over span (+/- radius)
  student-side signal (ONLINE / circular -- reported separately, not a driver):
    disagree    = 1 - p_student(token) at mapped teacher peak

Reports Pearson + Spearman of each signal vs offset and vs wrong@t_T, a
quartile table (signal bin -> mean offset / %wrong), and saves a bar figure.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import soundfile as sf
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel  # noqa: E402
import nemo.collections.asr as nemo_asr  # noqa: E402


def load_audio(path, sr=16000):
    wav, orig_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return torch.from_numpy(wav)


def load_student(config_path, ckpt_path, device):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if model_cfg.get("test_ds") is not None:
            del model_cfg.test_ds
    model = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def get_logprobs(model, wav, device):
    wav_t = wav.unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    T = int(enc_len[0].item())
    return log_probs[0, :T].float().cpu()  # (T, V)


def entropy_rows(logp):
    """Full-vocab entropy (nats) per frame from log-probs. logp: (T,V) tensor."""
    p = logp.exp()
    return -(p * logp).sum(-1)  # (T,)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort()
        r = np.empty(len(a), float)
        r[order] = np.arange(len(a), dtype=float)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        start = csum - cnt
        avg = (start + csum - 1) / 2.0
        return avg[inv]
    return pearson(rank(x), rank(y))


def quartile_table(sig, offset, wrong, nbins=4):
    sig = np.asarray(sig, float); offset = np.asarray(offset, float); wrong = np.asarray(wrong, float)
    qs = np.quantile(sig, np.linspace(0, 1, nbins + 1))
    qs[-1] += 1e-9
    rows = []
    for b in range(nbins):
        m = (sig >= qs[b]) & (sig < qs[b + 1])
        if m.sum() == 0:
            rows.append((qs[b], qs[b + 1], 0, float("nan"), float("nan")))
            continue
        rows.append((qs[b], qs[b + 1], int(m.sum()),
                     float(offset[m].mean()), float(100 * wrong[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train_clean_100.small_teacher.token_avg.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--ckpt", required=True, help="trained student checkpoint (comparison = span-KD)")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--search_radius", type=int, default=5)
    ap.add_argument("--spread_radius", type=int, default=3,
                    help="frames beyond segment used for teacher temporal-spread")
    ap.add_argument("--tag", default="span_kd", help="label for output figure/name")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[load] teacher: {args.teacher}")
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(device).eval()
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    print(f"[load] student: {args.ckpt}")
    student = load_student(args.config, args.ckpt, device)

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[analyze] {len(rows)} utts  blank_id={blank_id}")

    # per-segment collectors. targets = offset (integer peak dist),
    # soft_offset (continuous mass-centroid dist, higher dynamic range), wrong.
    S = {k: [] for k in ["offset", "soft_offset", "wrong", "seg_len", "H_peak",
                         "H_span", "H_agg", "unsharp", "blank_frac", "tspread",
                         "disagree"]}

    for ri, row in enumerate(rows):
        wav = load_audio(row["audio_filepath"])
        pt_path = row["teacher_token_avg_path"]
        if not os.path.isabs(pt_path):
            pt_path = os.path.join(args.data_dir, pt_path)
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
        seg_starts = pt["seg_starts"].tolist()
        seg_ends = pt["seg_ends"].tolist()
        avg_ids = pt["avg_ids"]
        avg_probs = pt["avg_probs"]  # (N, top_k) span-aggregated target dist

        t_logprobs = get_logprobs(teacher, wav, device)  # (T_t, V)
        s_logprobs = get_logprobs(student, wav, device)  # (T_s, V)
        T_t, T_s = t_logprobs.shape[0], s_logprobs.shape[0]
        scale = T_s / T_t if T_t > 0 else 1.0
        t_argmax = t_logprobs.argmax(-1)
        t_ent = entropy_rows(t_logprobs)  # (T_t,)

        for k in range(len(seg_starts)):
            tok_id = int(avg_ids[k, 0].item())
            if tok_id == blank_id:
                continue
            t_start = min(int(seg_starts[k]), T_t - 1)
            t_end = min(int(seg_ends[k]), T_t)
            if t_end <= t_start:
                continue

            teacher_seg = t_logprobs[t_start:t_end, tok_id]
            t_T = t_start + int(teacher_seg.argmax().item())
            s_T = min(int(round(t_T * scale)), T_s - 1)

            # --- misalignment target ---
            s_argmax = int(s_logprobs[s_T].argmax().item())
            wrong = int(s_argmax != tok_id)
            s_seg_start = max(0, min(int(t_start * scale), T_s - 1) - args.search_radius)
            s_seg_end = min(T_s, min(int(t_end * scale), T_s) + args.search_radius)
            if s_seg_end <= s_seg_start:
                continue
            s_window = s_logprobs[s_seg_start:s_seg_end, tok_id]
            t_S = s_seg_start + int(s_window.argmax().item())
            offset = abs(s_T - t_S)

            # --- teacher-side signals (offline) ---
            H_peak = float(t_ent[t_T])
            H_span = float(t_ent[t_start:t_end].mean())
            ap_k = np.asarray(avg_probs[k], float)
            ap_k = ap_k / max(ap_k.sum(), 1e-12)
            H_agg = float(-(ap_k * np.log(ap_k + 1e-12)).sum())
            unsharp = float(1.0 - t_logprobs[t_T].exp().max())
            blank_frac = float((t_argmax[t_start:t_end] == blank_id).float().mean())
            w0 = max(0, t_start - args.spread_radius)
            w1 = min(T_t, t_end + args.spread_radius)
            wmass = t_logprobs[w0:w1, tok_id].exp().numpy()
            idx = np.arange(w0, w1, dtype=float)
            wsum = wmass.sum()
            if wsum > 1e-12:
                mean_t = (idx * wmass).sum() / wsum          # teacher mass centroid
                tspread = float(np.sqrt(((idx - mean_t) ** 2 * wmass).sum() / wsum))
            else:
                mean_t = 0.5 * (t_start + t_end - 1)
                tspread = 0.0

            # --- continuous misalignment: teacher vs student mass centroid ---
            # student centroid over its search window, weighted by p_student(tok),
            # then compare in student-frame units (map teacher centroid by scale).
            smass = s_logprobs[s_seg_start:s_seg_end, tok_id].exp().numpy()
            sidx = np.arange(s_seg_start, s_seg_end, dtype=float)
            ssum = smass.sum()
            if ssum > 1e-12:
                mean_s = (sidx * smass).sum() / ssum
                soft_offset = float(abs(mean_t * scale - mean_s))
            else:
                soft_offset = float(abs(mean_t * scale - s_T))

            # --- student-side signal (circular; reference only) ---
            disagree = float(1.0 - s_logprobs[s_T].exp()[tok_id])

            S["offset"].append(offset); S["soft_offset"].append(soft_offset)
            S["wrong"].append(wrong)
            S["seg_len"].append(t_end - t_start)
            S["H_peak"].append(H_peak); S["H_span"].append(H_span); S["H_agg"].append(H_agg)
            S["unsharp"].append(unsharp); S["blank_frac"].append(blank_frac)
            S["tspread"].append(tspread); S["disagree"].append(disagree)

        if (ri + 1) % 50 == 0:
            print(f"  ...{ri + 1}/{len(rows)} utts, {len(S['offset'])} segs")

    n = len(S["offset"])
    if n == 0:
        print("no segments."); return
    offset = np.asarray(S["offset"], float)
    soft = np.asarray(S["soft_offset"], float)
    wrong = np.asarray(S["wrong"], float)
    print(f"\n=== Method-4 pre-check | student={args.tag} | segs={n} ===")
    print(f"offset      mean {offset.mean():.2f}  median {np.median(offset):.1f}  "
          f"std {offset.std():.2f}   (integer peak dist)")
    print(f"soft_offset mean {soft.mean():.2f}  median {np.median(soft):.2f}  "
          f"std {soft.std():.2f}   (continuous centroid dist = PRIMARY target)")
    print(f"wrong@t_T   {100*wrong.mean():.1f}%")

    signals = ["seg_len", "H_peak", "H_span", "H_agg", "unsharp",
               "blank_frac", "tspread", "disagree"]
    # PRIMARY target = soft_offset (continuous, real dynamic range).
    print(f"\n--- correlation: teacher-side signal vs misalignment ---")
    print(f"{'signal':>11} | {'r(soft)':>9} {'rho(soft)':>10} | "
          f"{'rho(offset)':>11} {'rho(wrong)':>11}   note")
    notes = {"disagree": "CIRCULAR (student-side)", "seg_len": "trivial baseline"}
    ranking = []
    for s in signals:
        v = np.asarray(S[s], float)
        r_s, rho_s = pearson(v, soft), spearman(v, soft)
        rho_o, rho_w = spearman(v, offset), spearman(v, wrong)
        print(f"{s:>11} | {r_s:>9.3f} {rho_s:>10.3f} | {rho_o:>11.3f} {rho_w:>11.3f}   "
              f"{notes.get(s, '')}")
        if s != "disagree":
            ranking.append((abs(rho_s), s, rho_s))
    ranking.sort(reverse=True)
    best = ranking[0][1] if ranking else "H_agg"
    print(f"\nbest OFFLINE predictor of soft_offset (|rho|): {best} "
          f"(rho={dict((s, r) for _, s, r in ranking)[best]:.3f})")

    # quartile tables: signal bin -> mean soft_offset (primary) + wrong%
    for s in dict.fromkeys([best, "H_agg", "tspread"]):
        print(f"\n--- {s} quartile -> misalignment (mean soft_offset) ---")
        print(f"{'bin':>4} {'lo':>8} {'hi':>8} {'n':>6} {'mean_soft':>10} {'wrong%':>8}")
        for i, (lo, hi, cnt, mo, wp) in enumerate(quartile_table(S[s], soft, wrong)):
            print(f"Q{i+1:>3} {lo:>8.3f} {hi:>8.3f} {cnt:>6} {mo:>10.2f} {wp:>7.1f}%")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_sigs = list(dict.fromkeys([best, "H_agg", "tspread", "seg_len"]))
        fig, axes = plt.subplots(1, len(plot_sigs), figsize=(4 * len(plot_sigs), 3.4))
        if len(plot_sigs) == 1:
            axes = [axes]
        for ax, s in zip(axes, plot_sigs):
            tab = quartile_table(S[s], soft, wrong)
            xs = [f"Q{i+1}" for i in range(len(tab))]
            ys = [t[3] for t in tab]
            ax.bar(xs, ys, color="#4C72B0")
            ax.set_title(f"{s}\nrho={spearman(S[s], soft):.3f}", fontsize=10)
            ax.set_ylabel("mean soft_offset (frames)")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Method-4 pre-check: teacher-side signal -> student misalignment "
                     f"({args.tag}, n={n})", fontsize=11)
        fig.tight_layout()
        os.makedirs(args.fig_dir, exist_ok=True)
        out = os.path.join(args.fig_dir, f"method4_precheck_{args.tag}.png")
        fig.savefig(out, dpi=140)
        print(f"\n[fig] {out}")
    except Exception as e:
        print(f"[fig] skipped: {e}")


if __name__ == "__main__":
    main()
