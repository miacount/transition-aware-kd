#!/usr/bin/env python3
"""Gate nonblank-logit-standardized dark knowledge before training.

WHERE is fixed to the teacher's transcript-constrained delta=6 FB occupancy.
WHAT sources are compared by whether their non-target distribution predicts
real student confusions, avoids transcript-neighbour leakage, and is stable to
a weak waveform perturbation.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_span_kd_targets import blank_penalty_logprobs, run_teacher  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


DEFAULT_NOKD = (
    ROOT / "nemo_experiments/student-no-kd/2026-06-24_04-38-47/"
    "checkpoints/student-no-kd.nemo"
)
DEFAULT_SPAN = (
    ROOT / "nemo_experiments/span-kd-teacher-d6-w20.0/2026-07-02_14-03-36/"
    "checkpoints/span-kd-teacher-d6-w20.0.nemo"
)


def normalize(x, axis=-1):
    return x / np.maximum(x.sum(axis=axis, keepdims=True), 1e-30)


def load_audio(path, expected_sr):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != expected_sr:
        raise ValueError(f"sample-rate mismatch: {sr} != {expected_sr}: {path}")
    return wav


def resample_prob(prob, frames):
    if prob.shape[0] == frames:
        return prob
    old = np.linspace(0.0, 1.0, prob.shape[0])
    new = np.linspace(0.0, 1.0, frames)
    out = np.stack([np.interp(new, old, prob[:, k]) for k in range(prob.shape[1])], 1)
    return normalize(out)


def posterior_sources(log_probs, blank):
    sources = {"raw": np.exp(log_probs)}
    scaled = log_probs / 2.0
    scaled -= scaled.max(axis=1, keepdims=True)
    sources["temperature_t2"] = normalize(np.exp(scaled))

    nb = log_probs[:, :blank].astype(np.float64)
    centred = nb - nb.mean(axis=1, keepdims=True)
    scale = nb.std(axis=1, keepdims=True)
    z = centred / np.maximum(scale, 1e-6)
    for tau in (0.5, 1.0, 2.0):
        zs = z / tau
        zs -= zs.max(axis=1, keepdims=True)
        q = np.zeros_like(log_probs, dtype=np.float64)
        q[:, :blank] = normalize(np.exp(zs))
        sources[f"nb_logit_std_t{tau:g}"] = q
    return sources


def pool(prob, gamma):
    weights = gamma / max(float(gamma.sum()), 1e-30)
    return (weights[:, None] * prob).sum(axis=0)


def without(q, excluded):
    out = q.copy()
    for idx in excluded:
        out[idx] = 0.0
    return normalize(out)


def entropy_neff(q):
    positive = q > 1e-30
    h = float(-(q[positive] * np.log(q[positive])).sum())
    return h, float(np.exp(h))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    kp, kq = p > 0, q > 0
    return 0.5 * float((p[kp] * np.log(p[kp] / m[kp])).sum()) + \
        0.5 * float((q[kq] * np.log(q[kq] / m[kq])).sum())


def top_set(q, k):
    k = min(k, q.size)
    return set(np.argpartition(q, -k)[-k:].tolist())


def weak_noise(wav, rng, snr_db=30.0):
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    noise_rms = rms / (10.0 ** (snr_db / 20.0))
    return (wav + rng.normal(0.0, noise_rms, wav.shape)).astype(np.float32)


def empty_source_acc():
    return {
        "gt_mass": [], "top1_gt": [], "entropy": [], "neff": [],
        "nt_entropy": [], "nt_neff": [], "nt_top1_neighbor": [],
        "nt_top8_neighbor_share": [], "top32_mass": [], "top64_mass": [],
        "nt_top1_ids": Counter(), "stable_jaccard32": [], "stable_js": [],
        "students": defaultdict(lambda: {
            "errors": 0, "ranks": [], "recall8": 0, "recall32": 0, "recall64": 0,
        }),
    }


def summarize(acc, tokenizer):
    out = {}
    for key in (
        "gt_mass", "entropy", "neff", "nt_entropy", "nt_neff",
        "nt_top8_neighbor_share", "top32_mass", "top64_mass",
        "stable_jaccard32", "stable_js",
    ):
        arr = np.asarray(acc[key], dtype=np.float64)
        out[key + "_mean"] = float(arr.mean()) if arr.size else None
        out[key + "_median"] = float(np.median(arr)) if arr.size else None
    n = max(len(acc["gt_mass"]), 1)
    out["n"] = len(acc["gt_mass"])
    out["top1_gt_pct"] = 100.0 * sum(acc["top1_gt"]) / n
    out["nt_top1_neighbor_pct"] = 100.0 * sum(acc["nt_top1_neighbor"]) / n
    common = acc["nt_top1_ids"].most_common(12)
    out["nt_top1_top12"] = [
        {"id": int(idx), "token": tokenizer.ids_to_tokens([int(idx)])[0], "count": int(count)}
        for idx, count in common
    ]
    out["nt_top1_top12_share_pct"] = 100.0 * sum(x[1] for x in common) / n
    out["students"] = {}
    for name, s in acc["students"].items():
        errors = max(s["errors"], 1)
        ranks = np.asarray(s["ranks"], dtype=np.float64)
        out["students"][name] = {
            "errors": int(s["errors"]),
            "mrr": float(np.mean(1.0 / ranks)) if ranks.size else None,
            "mean_rank": float(ranks.mean()) if ranks.size else None,
            "median_rank": float(np.median(ranks)) if ranks.size else None,
            "recall_at_8_pct": 100.0 * s["recall8"] / errors,
            "recall_at_32_pct": 100.0 * s["recall32"] / errors,
            "recall_at_64_pct": 100.0 * s["recall64"] / errors,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/test_clean.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--stability-limit", type=int, default=50)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer_1024"))
    ap.add_argument("--no-kd", default=str(DEFAULT_NOKD))
    ap.add_argument("--span-kd", default=str(DEFAULT_SPAN))
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--noise-snr", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "analysis/standardized_dark_clean.json"))
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    students = {
        "no_kd": nemo_asr.models.EncDecCTCModelBPE.restore_from(args.no_kd).to(args.device).eval(),
        "span_kd": nemo_asr.models.EncDecCTCModelBPE.restore_from(args.span_kd).to(args.device).eval(),
    }
    for model in students.values():
        model.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    with open(args.manifest) as f:
        rows = [json.loads(x) for x in f if x.strip()][:args.limit]

    acc = {name: empty_source_acc() for name in posterior_sources(
        np.zeros((1, blank + 1), dtype=np.float64), blank)}
    rng = np.random.default_rng(12345)
    total_tokens = 0

    for ri, row in enumerate(rows):
        wav = load_audio(row["audio_filepath"], sr)
        ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
        if not ids:
            continue
        lp = run_teacher(teacher, wav, args.device).astype(np.float64)
        where_lp = blank_penalty_logprobs(lp, blank, args.delta)
        gamma = ctc_forward_backward(where_lp, ids, blank)["token_gamma"].astype(np.float64)
        sources = posterior_sources(lp, blank)

        student_q = {}
        for student_name, model in students.items():
            slp = run_teacher(model, wav, args.device).astype(np.float64)
            sp = resample_prob(np.exp(slp), lp.shape[0])
            student_q[student_name] = [
                without(pool(sp, gamma[:, u]), {blank}) for u in range(len(ids))
            ]

        noisy_sources = None
        if ri < args.stability_limit:
            noisy = weak_noise(wav, rng, args.noise_snr)
            noisy_lp = run_teacher(teacher, noisy, args.device).astype(np.float64)
            if noisy_lp.shape[0] != lp.shape[0]:
                noisy_prob_sources = posterior_sources(noisy_lp, blank)
                noisy_sources = {
                    name: resample_prob(prob, lp.shape[0])
                    for name, prob in noisy_prob_sources.items()
                }
            else:
                noisy_sources = posterior_sources(noisy_lp, blank)

        for u, y in enumerate(ids):
            prev_y = ids[u - 1] if u else None
            next_y = ids[u + 1] if u + 1 < len(ids) else None
            neighbours = {x for x in (prev_y, next_y) if x is not None and x != y}
            for name, prob in sources.items():
                q = without(pool(prob, gamma[:, u]), {blank})
                nt = without(q, {blank, y})
                h, neff = entropy_neff(q)
                hn, neffn = entropy_neff(nt)
                order = np.argsort(nt)[::-1]
                a = acc[name]
                a["gt_mass"].append(float(q[y]))
                a["top1_gt"].append(int(int(q.argmax()) == y))
                a["entropy"].append(h)
                a["neff"].append(neff)
                a["nt_entropy"].append(hn)
                a["nt_neff"].append(neffn)
                a["nt_top1_neighbor"].append(int(int(order[0]) in neighbours))
                a["nt_top8_neighbor_share"].append(
                    sum(float(nt[v]) for v in neighbours if v in set(order[:8])))
                a["top32_mass"].append(float(nt[order[:32]].sum()))
                a["top64_mass"].append(float(nt[order[:64]].sum()))
                a["nt_top1_ids"][int(order[0])] += 1

                for student_name, sqs in student_q.items():
                    pred = int(sqs[u].argmax())
                    if pred == y:
                        continue
                    s = a["students"][student_name]
                    rank = int(np.where(order == pred)[0][0]) + 1
                    s["errors"] += 1
                    s["ranks"].append(rank)
                    s["recall8"] += int(rank <= 8)
                    s["recall32"] += int(rank <= 32)
                    s["recall64"] += int(rank <= 64)

                if noisy_sources is not None:
                    qn = without(pool(noisy_sources[name], gamma[:, u]), {blank})
                    ntn = without(qn, {blank, y})
                    sa, sb = top_set(nt, 32), top_set(ntn, 32)
                    a["stable_jaccard32"].append(len(sa & sb) / max(len(sa | sb), 1))
                    a["stable_js"].append(js_divergence(nt, ntn))
            total_tokens += 1
        if (ri + 1) % 25 == 0:
            print(f".. {ri + 1}/{len(rows)}", flush=True)

    result = {
        "config": vars(args), "utterances": len(rows), "tokens": total_tokens,
        "sources": {name: summarize(a, tokenizer) for name, a in acc.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nStandardized dark-knowledge diagnostic")
    header = (
        f"{'source':>20} {'GT':>6} {'nEff':>7} {'NTnEff':>8} {'neigh':>7} "
        f"{'noR@32':>8} {'spR@32':>8} {'Jacc32':>8} {'JSnoise':>8}"
    )
    print(header)
    for name, s in result["sources"].items():
        no = s["students"]["no_kd"]
        sp = s["students"]["span_kd"]
        print(
            f"{name:>20} {s['gt_mass_mean']:>6.3f} {s['neff_mean']:>7.2f} "
            f"{s['nt_neff_mean']:>8.2f} {s['nt_top1_neighbor_pct']:>6.1f}% "
            f"{no['recall_at_32_pct']:>7.1f}% {sp['recall_at_32_pct']:>7.1f}% "
            f"{s['stable_jaccard32_mean']:>8.3f} {s['stable_js_mean']:>8.4f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
