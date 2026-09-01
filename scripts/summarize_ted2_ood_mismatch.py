#!/usr/bin/env python
"""Summarize LS-trained Vanilla/AT-DKD on TED2 by natural spike offset."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from summarize_natural_mismatch import edit_distance


ROOT = Path("analysis/ted2_ood_lstrained_mismatch")
THRESHOLD = 0.6875
BOOTSTRAP_REPS = 5000


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def bootstrap_delta(ids, vanilla, atdkd, rng):
    values = np.empty(BOOTSTRAP_REPS)
    for rep in range(BOOTSTRAP_REPS):
        sampled = rng.integers(0, len(ids), len(ids))
        seed_delta = []
        for seed in (1, 2, 3):
            v_edits = a_edits = words = 0
            for index in sampled:
                utt = ids[int(index)]
                edit, count = vanilla[seed][utt]
                v_edits += edit
                words += count
                a_edits += atdkd[seed][utt][0]
            seed_delta.append(100.0 * (v_edits - a_edits) / words)
        values[rep] = statistics.mean(seed_delta)
    return values


def main():
    mismatch_rows = read_jsonl(ROOT / "per_utterance_no_kd.jsonl")
    mismatch = {row["utterance_id"]: row for row in mismatch_rows}
    groups = {
        "Small offset": [row["utterance_id"] for row in mismatch_rows
                         if row["mean_abs_offset"] <= THRESHOLD],
        "Large offset": [row["utterance_id"] for row in mismatch_rows
                         if row["mean_abs_offset"] > THRESHOLD],
        "Overall": [row["utterance_id"] for row in mismatch_rows],
    }

    counts = {"vanilla": {}, "atdkd": {}}
    for method in counts:
        for seed in (1, 2, 3):
            rows = read_jsonl(ROOT / "predictions" / f"{method}_s{seed}_ted2_test.jsonl")
            counts[method][seed] = {
                row["utterance_id"]: edit_distance(row["reference"], row["beam"])
                for row in rows
            }
            assert set(counts[method][seed]) == set(mismatch)

    rng = np.random.default_rng(20260828)
    results, bootstrap = [], {}
    for label, ids in groups.items():
        vanilla_wers, atdkd_wers, rerrs = [], [], []
        for seed in (1, 2, 3):
            v_edits = sum(counts["vanilla"][seed][utt][0] for utt in ids)
            a_edits = sum(counts["atdkd"][seed][utt][0] for utt in ids)
            words = sum(counts["vanilla"][seed][utt][1] for utt in ids)
            vanilla_wer = 100.0 * v_edits / words
            atdkd_wer = 100.0 * a_edits / words
            vanilla_wers.append(vanilla_wer)
            atdkd_wers.append(atdkd_wer)
            rerrs.append(100.0 * (vanilla_wer - atdkd_wer) / vanilla_wer)
        delta = [v - a for v, a in zip(vanilla_wers, atdkd_wers)]
        boot = bootstrap_delta(ids, counts["vanilla"], counts["atdkd"], rng)
        bootstrap[label] = boot
        rows = [mismatch[utt] for utt in ids]
        vm, vs = mean_sd(vanilla_wers)
        am, ass = mean_sd(atdkd_wers)
        dm, ds = mean_sd(delta)
        rm, rs = mean_sd(rerrs)
        low, high = np.quantile(boot, [0.025, 0.975])
        results.append({
            "group": label,
            "utterances": len(ids),
            "mean_abs_offset": statistics.mean(r["mean_abs_offset"] for r in rows),
            "recoverable_fraction": statistics.mean(r["recoverable_local_fraction"] for r in rows),
            "vanilla_wer_mean": vm,
            "vanilla_wer_sd": vs,
            "atdkd_wer_mean": am,
            "atdkd_wer_sd": ass,
            "delta_wer_mean": dm,
            "delta_wer_sd": ds,
            "delta_ci_low": float(low),
            "delta_ci_high": float(high),
            "rerr_mean": rm,
            "rerr_sd": rs,
        })

    contrast = bootstrap["Large offset"] - bootstrap["Small offset"]
    contrast_low, contrast_high = np.quantile(contrast, [0.025, 0.975])
    contrast_mean = float(contrast.mean())
    contrast_p = float((contrast <= 0).mean())

    csv_path = ROOT / "ted2_ood_offset_groups.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    md_path = ROOT / "ted2_ood_offset_groups.md"
    lines = [
        "# LibriSpeech-trained models on TED-LIUM2 test",
        "",
        f"Small/large groups were fixed before model evaluation at the median utterance-level "
        f"Teacher-No-KD mean absolute spike offset ({THRESHOLD:.4f} frames). WER is no-LM "
        "beam-16 mean +/- sample SD over three matched seeds. Positive Delta WER favors AT-DKD.",
        "",
        "| Group | Utts | Mean offset | Recoverable frac. | Vanilla WER | AT-DKD WER | Delta WER [95% CI] | RERR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['group']} | {row['utterances']} | {row['mean_abs_offset']:.3f} | "
            f"{row['recoverable_fraction']:.3f} | "
            f"{row['vanilla_wer_mean']:.2f} +/- {row['vanilla_wer_sd']:.2f} | "
            f"{row['atdkd_wer_mean']:.2f} +/- {row['atdkd_wer_sd']:.2f} | "
            f"{row['delta_wer_mean']:.2f} [{row['delta_ci_low']:.2f}, {row['delta_ci_high']:.2f}] | "
            f"{row['rerr_mean']:.2f} +/- {row['rerr_sd']:.2f} |"
        )
    lines += [
        "",
        f"Large-minus-small difference in AT-DKD WER reduction: {contrast_mean:.2f} pp "
        f"(95% bootstrap CI [{contrast_low:.2f}, {contrast_high:.2f}], "
        f"P[difference <= 0] = {contrast_p:.3f}).",
    ]
    md_path.write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    shown = results[:2]
    x = np.arange(2)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    axes[0].bar(x - width / 2, [r["vanilla_wer_mean"] for r in shown], width,
                yerr=[r["vanilla_wer_sd"] for r in shown], capsize=3,
                label="Vanilla KD", color="#d55e00")
    axes[0].bar(x + width / 2, [r["atdkd_wer_mean"] for r in shown], width,
                yerr=[r["atdkd_wer_sd"] for r in shown], capsize=3,
                label="AT-DKD", color="#009e73")
    axes[0].set_xticks(x, ["Small", "Large"])
    axes[0].set_ylabel("TED-LIUM2 test WER (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    delta = np.array([r["delta_wer_mean"] for r in shown])
    low = delta - np.array([r["delta_ci_low"] for r in shown])
    high = np.array([r["delta_ci_high"] for r in shown]) - delta
    axes[1].errorbar(x, delta, yerr=np.vstack([low, high]), marker="o",
                     capsize=4, color="#2f6fbb")
    axes[1].axhline(0, color="#555", linewidth=0.8)
    axes[1].set_xticks(x, ["Small", "Large"])
    axes[1].set_ylabel("WER reduction (Vanilla - AT-DKD, %p)")
    axes[1].grid(alpha=0.25)
    for suffix in (".png", ".pdf"):
        fig.savefig(ROOT / f"ted2_ood_offset_groups{suffix}", dpi=300, bbox_inches="tight")
    print(f"wrote {csv_path}, {md_path}, PNG, and PDF")


if __name__ == "__main__":
    main()
