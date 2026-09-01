#!/usr/bin/env python3
"""Diagnose rho-factorized Span-KD WHAT targets before training.

For a fixed blank-suppressed teacher FB occupancy (WHERE), construct

    q_rho(k|u) propto sum_t gamma(t,u) * m(t)^rho * r(k|t),

where m(t)=1-p(blank|t) and r(k|t)=p(k|t)/m(t), k != blank.
rho=1 is the current raw-posterior aggregation and rho=0 is per-frame
blank-conditional aggregation.  The report distinguishes useful softening from
adjacent-token leakage and very-low-nonblank-mass frame noise.  It also compares
FB with the no-FB type-global support on repeated token occurrences.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_span_kd_targets import blank_penalty_logprobs, run_teacher  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


MASS_BINS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.000001)


def load_audio(path, expected_sr):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != expected_sr:
        raise ValueError(f"sample-rate mismatch for {path}: {sr} != {expected_sr}")
    return wav


def normalize(x, axis=-1):
    return x / np.maximum(x.sum(axis=axis, keepdims=True), 1e-30)


def target_for(r, mass, support, rho, blank):
    weights = support * np.power(np.maximum(mass, 1e-30), rho)
    weights = weights / max(float(weights.sum()), 1e-30)
    q = (weights[:, None] * r).sum(0)
    q[blank] = 0.0
    return normalize(q), weights


def local_target_for(r, mass, support, blank, offsets):
    """Use conditional WHAT only at selected offsets from the FB peak."""
    centre = int(np.argmax(support))
    rho_t = np.ones_like(mass, dtype=np.float64)
    for offset in offsets:
        t = centre + offset
        if 0 <= t < len(rho_t):
            rho_t[t] = 0.0
    weights = support * np.power(np.maximum(mass, 1e-30), rho_t)
    weights = weights / max(float(weights.sum()), 1e-30)
    q = (weights[:, None] * r).sum(0)
    q[blank] = 0.0
    return normalize(q), weights


def js_divergence(p, q):
    mid = 0.5 * (p + q)
    keep_p = p > 0
    keep_q = q > 0
    return 0.5 * float((p[keep_p] * np.log(p[keep_p] / mid[keep_p])).sum()) + \
        0.5 * float((q[keep_q] * np.log(q[keep_q] / mid[keep_q])).sum())


def empty_acc():
    return {
        "top1": [], "gt_mass": [], "nonself": [], "entropy": [], "neff": [],
        "top1_gt": [], "runner_neighbor": [], "runner_prev": [], "runner_next": [],
        "weighted_frame_gt": [], "weight_by_mass_bin": defaultdict(float),
        "runner_tokens": Counter(), "n": 0,
    }


def add_target(acc, q, weights, r, mass, y, prev_y, next_y, tokenizer, blank):
    order = np.argsort(q)[::-1]
    top1, runner = int(order[0]), int(order[1])
    positive = q > 1e-30
    entropy = float(-(q[positive] * np.log(q[positive])).sum())
    acc["top1"].append(float(q[top1]))
    acc["gt_mass"].append(float(q[y]))
    acc["nonself"].append(float(1.0 - q[y]))
    acc["entropy"].append(entropy)
    acc["neff"].append(float(np.exp(entropy)))
    acc["top1_gt"].append(int(top1 == y))
    acc["runner_prev"].append(int(prev_y is not None and runner == prev_y))
    acc["runner_next"].append(int(next_y is not None and runner == next_y))
    acc["runner_neighbor"].append(int(runner in {x for x in (prev_y, next_y) if x is not None}))
    frame_argmax = r.argmax(1)
    acc["weighted_frame_gt"].append(float(weights[frame_argmax == y].sum()))
    for bi in range(len(MASS_BINS) - 1):
        mask = (mass >= MASS_BINS[bi]) & (mass < MASS_BINS[bi + 1])
        acc["weight_by_mass_bin"][str(bi)] += float(weights[mask].sum())
    try:
        runner_str = tokenizer.ids_to_tokens([runner])[0]
    except Exception:
        runner_str = str(runner)
    acc["runner_tokens"][runner_str] += 1
    acc["n"] += 1


def summarize(acc):
    n = max(acc["n"], 1)
    result = {"n": acc["n"]}
    for key in ("top1", "gt_mass", "nonself", "entropy", "neff", "weighted_frame_gt"):
        vals = np.asarray(acc[key], dtype=np.float64)
        result[key + "_mean"] = float(vals.mean()) if vals.size else None
        result[key + "_median"] = float(np.median(vals)) if vals.size else None
    for key in ("top1_gt", "runner_neighbor", "runner_prev", "runner_next"):
        result[key + "_pct"] = 100.0 * float(sum(acc[key])) / n
    result["weight_by_mass_bin_pct"] = {
        f"[{MASS_BINS[i]:g},{MASS_BINS[i+1]:g})":
            100.0 * acc["weight_by_mass_bin"].get(str(i), 0.0) / n
        for i in range(len(MASS_BINS) - 1)
    }
    result["runner_tokens_top12"] = acc["runner_tokens"].most_common(12)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/train_clean_100.json"))
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer_1024"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--delta-where", type=float, default=6.0)
    ap.add_argument("--rhos", default="1,0.5,0.25,0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "analysis/conditional_span_target_diagnostic.json"))
    args = ap.parse_args()
    rhos = [float(x) for x in args.rhos.split(",")]

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    with open(args.manifest) as f:
        rows = [json.loads(line) for line in f if line.strip()][:args.limit]

    acc = {rho: empty_acc() for rho in rhos}
    local_specs = {
        "local_left1": (-1, 0),
        "local_right1": (0, 1),
        "local_sym1": (-1, 0, 1),
    }
    local_acc = {name: empty_acc() for name in local_specs}
    fb_nofb = {
        rho: {"repeat_l1": [], "unique_l1": [], "repeat_js_between_occ_fb": [],
              "repeat_js_between_occ_nofb": [], "repeat_nonself_fb": [],
              "repeat_nonself_nofb": []}
        for rho in rhos
    }

    for ri, row in enumerate(rows):
        ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
        if not ids:
            continue
        wav = load_audio(row["audio_filepath"], sr)
        lp = run_teacher(teacher, wav, args.device)
        prob = np.exp(lp).astype(np.float64)
        mass = np.maximum(1.0 - prob[:, blank], 1e-30)
        cond = prob / mass[:, None]
        cond[:, blank] = 0.0
        cond = normalize(cond)
        where_lp = blank_penalty_logprobs(lp, blank, args.delta_where)
        gamma = ctc_forward_backward(where_lp, ids, blank)["token_gamma"].astype(np.float64)
        nofb = np.exp(where_lp[:, ids]).astype(np.float64)
        counts = Counter(ids)

        for rho in rhos:
            q_fb = []
            q_nf = []
            for u, y in enumerate(ids):
                q, weights = target_for(cond, mass, gamma[:, u], rho, blank)
                q0, _ = target_for(cond, mass, nofb[:, u], rho, blank)
                q_fb.append(q)
                q_nf.append(q0)
                prev_y = ids[u - 1] if u > 0 else None
                next_y = ids[u + 1] if u + 1 < len(ids) else None
                add_target(acc[rho], q, weights, cond, mass, y, prev_y, next_y, tokenizer, blank)
                l1 = float(np.abs(q - q0).sum())
                key = "repeat_l1" if counts[y] > 1 else "unique_l1"
                fb_nofb[rho][key].append(l1)
                if counts[y] > 1:
                    fb_nofb[rho]["repeat_nonself_fb"].append(float(1.0 - q[y]))
                    fb_nofb[rho]["repeat_nonself_nofb"].append(float(1.0 - q0[y]))
            q_fb = np.stack(q_fb)
            q_nf = np.stack(q_nf)
            positions = defaultdict(list)
            for u, y in enumerate(ids):
                positions[y].append(u)
            for pos in positions.values():
                if len(pos) < 2:
                    continue
                for a, b in zip(pos[:-1], pos[1:]):
                    fb_nofb[rho]["repeat_js_between_occ_fb"].append(js_divergence(q_fb[a], q_fb[b]))
                    fb_nofb[rho]["repeat_js_between_occ_nofb"].append(js_divergence(q_nf[a], q_nf[b]))

        for name, offsets in local_specs.items():
            for u, y in enumerate(ids):
                q, weights = local_target_for(cond, mass, gamma[:, u], blank, offsets)
                prev_y = ids[u - 1] if u > 0 else None
                next_y = ids[u + 1] if u + 1 < len(ids) else None
                add_target(local_acc[name], q, weights, cond, mass, y, prev_y, next_y,
                           tokenizer, blank)
        if (ri + 1) % 25 == 0:
            print(f".. {ri + 1}/{len(rows)}", flush=True)

    output = {
        "config": vars(args),
        "mass_bins": list(MASS_BINS),
        "rho": {},
        "local": {},
        "fb_vs_nofb": {},
    }
    print(f"\nConditional Span target diagnostic: {len(rows)} utterances, delta_where={args.delta_where}")
    print(f"{'rho':>6} {'n':>7} {'top1':>8} {'GTmass':>8} {'n_eff':>8} {'top1=GT':>9} "
          f"{'r2=neigh':>10} {'frameGTw':>9}")
    print("-" * 78)
    for rho in rhos:
        s = summarize(acc[rho])
        output["rho"][str(rho)] = s
        print(f"{rho:>6g} {s['n']:>7} {s['top1_mean']:>8.3f} {s['gt_mass_mean']:>8.3f} "
              f"{s['neff_mean']:>8.2f} {s['top1_gt_pct']:>8.2f}% "
              f"{s['runner_neighbor_pct']:>9.2f}% {s['weighted_frame_gt_mean']:>8.3f}")
        print("       weight by p(nonblank): " + ", ".join(
            f"{k}={v:.1f}%" for k, v in s["weight_by_mass_bin_pct"].items()))

        fs = {}
        for key, vals in fb_nofb[rho].items():
            arr = np.asarray(vals, dtype=np.float64)
            fs[key + "_mean"] = float(arr.mean()) if arr.size else None
            fs[key + "_median"] = float(np.median(arr)) if arr.size else None
            fs[key + "_n"] = int(arr.size)
        output["fb_vs_nofb"][str(rho)] = fs
        print(f"       FB/noFB: repeat L1={fs['repeat_l1_mean']:.4f}, "
              f"unique L1={fs['unique_l1_mean']:.4f}, "
              f"repeat nonself={fs['repeat_nonself_fb_mean']:.4f}/{fs['repeat_nonself_nofb_mean']:.4f}, "
              f"occ-JS={fs['repeat_js_between_occ_fb_mean']:.4f}/{fs['repeat_js_between_occ_nofb_mean']:.4f}")

    print("\nLocal conditional variants (rho=0 at FB-peak offsets; rho=1 elsewhere)")
    print(f"{'variant':>14} {'n':>7} {'top1':>8} {'GTmass':>8} {'n_eff':>8} {'top1=GT':>9} "
          f"{'r2=neigh':>10} {'frameGTw':>9}")
    print("-" * 88)
    for name in local_specs:
        s = summarize(local_acc[name])
        s["offsets"] = list(local_specs[name])
        output["local"][name] = s
        print(f"{name:>14} {s['n']:>7} {s['top1_mean']:>8.3f} {s['gt_mass_mean']:>8.3f} "
              f"{s['neff_mean']:>8.2f} {s['top1_gt_pct']:>8.2f}% "
              f"{s['runner_neighbor_pct']:>9.2f}% {s['weighted_frame_gt_mean']:>8.3f}")
        print("                 weight by p(nonblank): " + ", ".join(
            f"{k}={v:.1f}%" for k, v in s["weight_by_mass_bin_pct"].items()))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
