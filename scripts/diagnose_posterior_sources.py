#!/usr/bin/env python3
"""Compare alternative WHAT posterior sources on a fixed teacher FB support."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_span_kd_targets import blank_penalty_logprobs, run_teacher  # noqa: E402
from diagnose_conditional_span_targets import add_target, empty_acc, normalize, summarize  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


MODEL_SOURCES = {
    "learned_depeak_d025": "nemo_experiments/teacher_depeak_d0.25/teacher_depeak.nemo",
    "learned_depeak_d05": "nemo_experiments/teacher_depeak_d0.5/teacher_depeak.nemo",
    "learned_depeak_d10": "nemo_experiments/teacher_depeak_d1.0/teacher_depeak.nemo",
    "cr_ctc": "nemo_experiments/cr_ctc_w02_v5/2026-07-02_01-42-26/checkpoints/cr_ctc_w02_v5.nemo",
    "span_kd_student": "nemo_experiments/span-kd-teacher-d6-w20.0/2026-07-02_14-03-36/checkpoints/span-kd-teacher-d6-w20.0.nemo",
}


def audio(path, sr):
    wav, got = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if got != sr:
        raise ValueError(f"sample-rate mismatch: {got} != {sr}: {path}")
    return wav


def resample_prob(prob, frames):
    if prob.shape[0] == frames:
        return prob
    old = np.linspace(0.0, 1.0, prob.shape[0])
    new = np.linspace(0.0, 1.0, frames)
    out = np.stack([np.interp(new, old, prob[:, k]) for k in range(prob.shape[1])], 1)
    return normalize(out)


def time_smooth(prob, strength):
    left = np.concatenate([prob[:1], prob[:-1]], 0)
    right = np.concatenate([prob[1:], prob[-1:]], 0)
    return 0.5 * strength * left + (1.0 - strength) * prob + 0.5 * strength * right


def target(prob, support, blank):
    w = support / max(float(support.sum()), 1e-30)
    q = (w[:, None] * prob).sum(0)
    q[blank] = 0.0
    return normalize(q), w


def add_source(acc, acc_topk, comparisons, prob, rec, tokenizer, blank, top_k=8):
    mass = np.maximum(1.0 - prob[:, blank], 1e-30)
    cond = prob / mass[:, None]
    cond[:, blank] = 0.0
    cond = normalize(cond)
    ids = rec["ids"]
    for u, y in enumerate(ids):
        q, w = target(prob, rec["gamma"][:, u], blank)
        add_target(acc, q, w, cond, mass, y, ids[u - 1] if u else None,
                   ids[u + 1] if u + 1 < len(ids) else None, tokenizer, blank)
        keep = np.argpartition(q, -top_k)[-top_k:]
        q_topk = np.zeros_like(q)
        q_topk[keep] = q[keep]
        q_topk = normalize(q_topk)
        add_target(acc_topk, q_topk, w, cond, mass, y, ids[u - 1] if u else None,
                   ids[u + 1] if u + 1 < len(ids) else None, tokenizer, blank)
        comparisons.append(float(np.abs(q - rec["q_base"][u]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/train_clean_100.json"))
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer_1024"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--delta-where", type=float, default=6.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "analysis/posterior_source_diagnostic.json"))
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    with open(args.manifest) as f:
        rows = [json.loads(line) for line in f if line.strip()][:args.limit]

    records = []
    for i, row in enumerate(rows):
        wav = audio(row["audio_filepath"], sr)
        ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
        if not ids:
            continue
        lp = run_teacher(teacher, wav, args.device).astype(np.float64)
        prob = np.exp(lp)
        where = blank_penalty_logprobs(lp, blank, args.delta_where)
        gamma = ctc_forward_backward(where, ids, blank)["token_gamma"].astype(np.float64)
        q_base = np.stack([target(prob, gamma[:, u], blank)[0] for u in range(len(ids))])
        records.append({"wav": wav, "ids": ids, "lp": lp, "prob": prob,
                        "gamma": gamma, "q_base": q_base})
        if (i + 1) % 25 == 0:
            print(f"base .. {i+1}/{len(rows)}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    acc = {}
    acc_topk = {}
    diffs = {}

    def evaluate_cached(name, transform):
        acc[name], acc_topk[name], diffs[name] = empty_acc(), empty_acc(), []
        for rec in records:
            add_source(acc[name], acc_topk[name], diffs[name], transform(rec), rec,
                       tokenizer, blank)

    evaluate_cached("raw_teacher", lambda rec: rec["prob"])
    for temp in (1.5, 2.0):
        evaluate_cached(f"temperature_T{temp:g}",
                        lambda rec, t=temp: normalize(np.exp(rec["lp"] / t)))
    for strength in (0.25, 0.5):
        evaluate_cached(f"time_smooth_s{strength:g}",
                        lambda rec, s=strength: time_smooth(rec["prob"], s))

    # Learned/model posterior sources; WHERE always remains original teacher FB.
    source_probs = {}
    for name, rel_path in MODEL_SOURCES.items():
        path = ROOT / rel_path
        if not path.exists():
            print(f"skip missing {name}: {path}")
            continue
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(path), map_location=args.device)
        model = model.to(args.device).eval()
        model.freeze()
        acc[name], acc_topk[name], diffs[name], source_probs[name] = empty_acc(), empty_acc(), [], []
        for i, rec in enumerate(records):
            lp = run_teacher(model, rec["wav"], args.device).astype(np.float64)
            prob = resample_prob(np.exp(lp), rec["prob"].shape[0])
            source_probs[name].append(prob)
            add_source(acc[name], acc_topk[name], diffs[name], prob, rec, tokenizer, blank)
        del model
        torch.cuda.empty_cache()
        print(f"source done: {name}", flush=True)

    # A conservative ensemble: retain the accurate original teacher and add the
    # learned de-peaked source that should contain genuinely broadened emissions.
    for src in ("learned_depeak_d025", "learned_depeak_d05", "learned_depeak_d10"):
        if src not in source_probs:
            continue
        name = "ensemble_raw_" + src
        acc[name], acc_topk[name], diffs[name] = empty_acc(), empty_acc(), []
        for rec, other in zip(records, source_probs[src]):
            prob = 0.5 * rec["prob"] + 0.5 * other
            add_source(acc[name], acc_topk[name], diffs[name], prob, rec, tokenizer, blank)

    output = {"config": vars(args), "sources": {}}
    print("\nFixed-WHERE posterior source diagnostic")
    print(f"{'source':>27} {'GTmass':>7} {'nEff':>6} {'K8GT':>7} {'K8nEff':>7} "
          f"{'top1GT':>8} {'r2neigh':>9} {'L1/base':>8}")
    print("-" * 96)
    for name, a in acc.items():
        s = summarize(a)
        sk = summarize(acc_topk[name])
        d = np.asarray(diffs[name])
        s["l1_vs_raw_mean"] = float(d.mean())
        s["l1_vs_raw_median"] = float(np.median(d))
        s["top8"] = sk
        output["sources"][name] = s
        print(f"{name:>27} {s['gt_mass_mean']:>7.3f} {s['neff_mean']:>6.2f} "
              f"{sk['gt_mass_mean']:>7.3f} {sk['neff_mean']:>7.2f} "
              f"{sk['top1_gt_pct']:>7.2f}% {sk['runner_neighbor_pct']:>8.2f}% "
              f"{s['l1_vs_raw_mean']:>8.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
