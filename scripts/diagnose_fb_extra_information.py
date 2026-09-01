#!/usr/bin/env python3
"""Gate counterfactual-delta and GT-vs-hypothesis FB signals before training."""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_span_kd_targets import blank_penalty_logprobs, run_teacher  # noqa: E402
def edit_align(reference, hypothesis):
    """Return a minimum-edit alignment without an experiment-builder dependency."""
    n, m = len(reference), len(hypothesis)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1])
            cost[i, j] = min(substitution, cost[i - 1, j] + 1, cost[i, j - 1] + 1)

    operations = []
    i, j = n, m
    while i or j:
        if i and j:
            step = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if cost[i, j] == cost[i - 1, j - 1] + step:
                kind = "correct" if step == 0 else "substitution"
                operations.append((kind, i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i and cost[i, j] == cost[i - 1, j] + 1:
            operations.append(("deletion", i - 1, None))
            i -= 1
        else:
            operations.append(("insertion", None, j - 1))
            j -= 1
    return operations[::-1]

from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


def load_audio(path, sr):
    wav, got = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if got != sr:
        raise ValueError(f"sample-rate mismatch {got} != {sr}: {path}")
    return wav


def collapse_greedy(lp, blank):
    out, prev = [], None
    for value in lp.argmax(1):
        value = int(value)
        if value != prev and value != blank:
            out.append(value)
        prev = value
    return out


def norm_col(x):
    return x / np.maximum(x.sum(0, keepdims=True), 1e-30)


def resample_cols(x, frames):
    if x.shape[0] == frames:
        return x
    old = np.linspace(0.0, 1.0, x.shape[0])
    new = np.linspace(0.0, 1.0, frames)
    return np.stack([np.interp(new, old, x[:, u]) for u in range(x.shape[1])], 1)


def overlap(p, q):
    p = p / max(float(p.sum()), 1e-30)
    q = q / max(float(q.sum()), 1e-30)
    return float(np.minimum(p, q).sum())


def weighted_component_stats(component, ids, cond, no_peak, span_peak, neighbors, acc):
    for u, y in enumerate(ids):
        w = component[:, u]
        mass = float(w.sum())
        if mass <= 1e-12:
            continue
        own = float((w * cond[:, y]).sum() / mass)
        nbr_ids = neighbors[u]
        nbr = float((w[:, None] * cond[:, nbr_ids]).sum() / mass) if nbr_ids else 0.0
        peak = float(w.max())
        acc["mass"].append(mass)
        acc["own"].append(own)
        acc["neighbor"].append(nbr)
        acc["hit_no"].append(int(w[no_peak[u]] > 0.01 * peak))
        acc["hit_span"].append(int(w[span_peak[u]] > 0.01 * peak))


def summarize_component(acc):
    return {
        "n": len(acc["mass"]),
        "mass_per_token": float(np.mean(acc["mass"])) if acc["mass"] else 0.0,
        "own_identity": float(np.average(acc["own"], weights=acc["mass"])) if acc["mass"] else 0.0,
        "neighbor_identity": float(np.average(acc["neighbor"], weights=acc["mass"])) if acc["mass"] else 0.0,
        "no_kd_peak_hit_pct": 100.0 * float(np.mean(acc["hit_no"])) if acc["mass"] else 0.0,
        "span_peak_hit_pct": 100.0 * float(np.mean(acc["hit_span"])) if acc["mass"] else 0.0,
    }


def quartiles(values, metrics):
    x = np.asarray(values)
    edges = np.quantile(x, [0, .25, .5, .75, 1.0])
    edges[-1] += 1e-9
    out = []
    for i in range(4):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        row = {"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(mask.sum())}
        for name, vals in metrics.items():
            row[name] = float(np.asarray(vals)[mask].mean()) if mask.any() else None
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/test_clean.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer_1024"))
    ap.add_argument("--no-kd", default=str(ROOT / "nemo_experiments/student-no-kd/2026-06-24_04-38-47/checkpoints/student-no-kd.nemo"))
    ap.add_argument("--span-kd", default=str(ROOT / "nemo_experiments/span-kd-teacher-d6-w20.0/2026-07-02_14-03-36/checkpoints/span-kd-teacher-d6-w20.0.nemo"))
    ap.add_argument("--deltas", default="0,2,4,6,9")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "analysis/fb_extra_information.json"))
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    deltas = [float(x) for x in args.deltas.split(",")]
    assert deltas == sorted(deltas) and 0.0 in deltas and 6.0 in deltas
    tok = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    no_kd = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.no_kd, map_location=args.device).to(args.device).eval()
    span_kd = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.span_kd, map_location=args.device).to(args.device).eval()
    for model in (teacher, no_kd, span_kd):
        model.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    rows = [json.loads(x) for x in open(args.manifest) if x.strip()][:args.limit]

    component_acc = {"core_d0": defaultdict(list)}
    for a, b in zip(deltas[:-1], deltas[1:]):
        component_acc[f"recruit_{a:g}_{b:g}"] = defaultdict(list)
    agreement = defaultdict(list)
    counts = defaultdict(int)

    for ri, row in enumerate(rows):
        ids = [int(x) for x in tok.text_to_ids(row["text"]) if int(x) != blank]
        if not ids:
            continue
        wav = load_audio(row["audio_filepath"], sr)
        t_lp = run_teacher(teacher, wav, args.device).astype(np.float64)
        n_lp = run_teacher(no_kd, wav, args.device).astype(np.float64)
        s_lp = run_teacher(span_kd, wav, args.device).astype(np.float64)
        gammas = {
            d: ctc_forward_backward(blank_penalty_logprobs(t_lp, blank, d), ids, blank)["token_gamma"]
            for d in deltas
        }
        n_gamma = resample_cols(ctc_forward_backward(n_lp, ids, blank)["token_gamma"], t_lp.shape[0])
        s_gamma = resample_cols(ctc_forward_backward(s_lp, ids, blank)["token_gamma"], t_lp.shape[0])
        no_peak = n_gamma.argmax(0)
        span_peak = s_gamma.argmax(0)
        prob = np.exp(t_lp)
        nonblank = np.maximum(1.0 - prob[:, blank], 1e-30)
        cond = prob / nonblank[:, None]
        cond[:, blank] = 0.0
        cond /= np.maximum(cond.sum(1, keepdims=True), 1e-30)
        neighbors = []
        for u, y in enumerate(ids):
            neighbors.append(list({v for v in (ids[u-1:u] + ids[u+1:u+2]) if v != y}))

        weighted_component_stats(gammas[0.0], ids, cond, no_peak, span_peak,
                                 neighbors, component_acc["core_d0"])
        for a, b in zip(deltas[:-1], deltas[1:]):
            inc = np.maximum(gammas[b] - gammas[a], 0.0)
            weighted_component_stats(inc, ids, cond, no_peak, span_peak, neighbors,
                                     component_acc[f"recruit_{a:g}_{b:g}"])

        hyp = collapse_greedy(t_lp, blank)
        ops = edit_align(ids, hyp)
        hyp_gamma = None
        if hyp:
            hyp_gamma = ctc_forward_backward(blank_penalty_logprobs(t_lp, blank, 6.0), hyp, blank)["token_gamma"]
        matched = {}
        for kind, gi, hi in ops:
            counts[kind] += 1
            if kind == "correct":
                matched[gi] = hi
        for u in range(len(ids)):
            counts["gt_tokens"] += 1
            agreement["matched"].append(int(u in matched))
            no_off = abs(int(no_peak[u]) - int(gammas[6.0][:, u].argmax()))
            sp_off = abs(int(span_peak[u]) - int(gammas[6.0][:, u].argmax()))
            if u not in matched:
                continue
            a = overlap(gammas[6.0][:, u], hyp_gamma[:, matched[u]])
            agreement["value"].append(a)
            agreement["no_offset"].append(no_off)
            agreement["span_offset"].append(sp_off)
            agreement["no_wrong_peak"].append(int(int(no_peak[u]) != int(gammas[6.0][:, u].argmax())))
            agreement["span_wrong_peak"].append(int(int(span_peak[u]) != int(gammas[6.0][:, u].argmax())))
        if (ri + 1) % 25 == 0:
            print(f".. {ri+1}/{len(rows)}", flush=True)

    components = {name: summarize_component(acc) for name, acc in component_acc.items()}
    matched_pct = 100.0 * sum(agreement["matched"]) / max(len(agreement["matched"]), 1)
    aq = quartiles(agreement["value"], {
        "no_offset": agreement["no_offset"], "span_offset": agreement["span_offset"],
        "no_wrong_peak": agreement["no_wrong_peak"], "span_wrong_peak": agreement["span_wrong_peak"],
    })
    output = {"config": vars(args), "counts": dict(counts), "components": components,
              "hypothesis_exact_match_pct": matched_pct,
              "agreement_mean": float(np.mean(agreement["value"])),
              "agreement_median": float(np.median(agreement["value"])),
              "agreement_quartiles": aq}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print("\nDelta-response components")
    print(f"{'component':>15} {'mass/tok':>9} {'own':>8} {'neighbor':>9} {'hit noKD':>10} {'hit span':>10}")
    for name, x in components.items():
        print(f"{name:>15} {x['mass_per_token']:>9.4f} {x['own_identity']:>8.3f} "
              f"{x['neighbor_identity']:>9.4f} {x['no_kd_peak_hit_pct']:>9.1f}% "
              f"{x['span_peak_hit_pct']:>9.1f}%")
    print(f"\nGT/HYP exact occurrence match: {matched_pct:.2f}%  "
          f"agreement mean/median={output['agreement_mean']:.3f}/{output['agreement_median']:.3f}")
    for i, row in enumerate(aq, 1):
        print(f"  Q{i} [{row['lo']:.3f},{row['hi']:.3f}) n={row['n']} "
              f"no_off={row['no_offset']:.3f} span_off={row['span_offset']:.3f} "
              f"no_wrong={100*row['no_wrong_peak']:.1f}% span_wrong={100*row['span_wrong_peak']:.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
