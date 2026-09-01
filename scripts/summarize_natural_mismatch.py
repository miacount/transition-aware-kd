#!/usr/bin/env python
"""Stratify matched-seed WER by precomputed natural mismatch burden."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


BOUNDARIES = (0.2, 0.2642593797440902, 1.0 / 3.0)
BIN_LABELS = (
    "Q1: <=0.200",
    "Q2: 0.200-0.264",
    "Q3: 0.264-0.333",
    "Q4: >0.333",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bin_index(value: float) -> int:
    for index, upper in enumerate(BOUNDARIES):
        if value <= upper:
            return index
    return 3


def edit_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = reference.split()
    hyp = hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1], len(ref)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def bootstrap_delta(
    ids: list[str],
    vanilla_counts: dict[int, dict[str, tuple[int, int]]],
    atdkd_counts: dict[int, dict[str, tuple[int, int]]],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(ids)
    deltas = np.empty(repetitions)
    for rep in range(repetitions):
        sampled = rng.integers(0, n, n)
        per_seed = []
        for seed in (1, 2, 3):
            v_edits = v_words = a_edits = a_words = 0
            for index in sampled:
                utt = ids[int(index)]
                e, w = vanilla_counts[seed][utt]
                v_edits += e
                v_words += w
                e, w = atdkd_counts[seed][utt]
                a_edits += e
                a_words += w
            per_seed.append(100.0 * (v_edits / v_words - a_edits / a_words))
        deltas[rep] = statistics.mean(per_seed)
    return tuple(float(x) for x in np.quantile(deltas, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mismatch",
        type=Path,
        default=Path("analysis/natural_mismatch_test_clean/per_utterance_no_kd.jsonl"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("analysis/natural_mismatch_test_clean/predictions"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("analysis/natural_mismatch_test_clean/natural_mismatch_wer"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    mismatch_rows = read_jsonl(args.mismatch)
    mismatch = {row["utterance_id"]: row for row in mismatch_rows}
    bins = {index: [] for index in range(4)}
    for row in mismatch_rows:
        bins[bin_index(row["recoverable_local_fraction"])].append(row["utterance_id"])

    counts: dict[str, dict[int, dict[str, tuple[int, int]]]] = {
        "vanilla": {},
        "atdkd": {},
    }
    for method in counts:
        for seed in (1, 2, 3):
            rows = read_jsonl(args.predictions_dir / f"{method}_s{seed}_test_clean.jsonl")
            if len(rows) != len(mismatch_rows):
                raise RuntimeError(f"{method} seed {seed}: {len(rows)} != {len(mismatch_rows)}")
            counts[method][seed] = {
                row["utterance_id"]: edit_distance(row["reference"], row["beam"])
                for row in rows
            }
            if set(counts[method][seed]) != set(mismatch):
                raise RuntimeError(f"{method} seed {seed}: utterance IDs do not match mismatch file")

    rng = np.random.default_rng(args.seed)
    results = []
    for index in range(4):
        ids = bins[index]
        vanilla_wers, atdkd_wers, rerrs = [], [], []
        ref_words = None
        for seed in (1, 2, 3):
            v_edits = sum(counts["vanilla"][seed][utt][0] for utt in ids)
            v_words = sum(counts["vanilla"][seed][utt][1] for utt in ids)
            a_edits = sum(counts["atdkd"][seed][utt][0] for utt in ids)
            a_words = sum(counts["atdkd"][seed][utt][1] for utt in ids)
            if v_words != a_words:
                raise RuntimeError("reference denominators differ between matched models")
            ref_words = v_words
            vanilla_wer = 100.0 * v_edits / v_words
            atdkd_wer = 100.0 * a_edits / a_words
            vanilla_wers.append(vanilla_wer)
            atdkd_wers.append(atdkd_wer)
            rerrs.append(100.0 * (vanilla_wer - atdkd_wer) / vanilla_wer)
        vanilla_mean, vanilla_sd = mean_sd(vanilla_wers)
        atdkd_mean, atdkd_sd = mean_sd(atdkd_wers)
        rerr_mean, rerr_sd = mean_sd(rerrs)
        delta_per_seed = [v - a for v, a in zip(vanilla_wers, atdkd_wers)]
        delta_mean, delta_sd = mean_sd(delta_per_seed)
        ci_low, ci_high = bootstrap_delta(
            ids, counts["vanilla"], counts["atdkd"], args.bootstrap, rng
        )
        rows = [mismatch[utt] for utt in ids]
        results.append({
            "bin": index + 1,
            "label": BIN_LABELS[index],
            "utterances": len(ids),
            "reference_words": ref_words,
            "recoverable_fraction_mean": statistics.mean(
                row["recoverable_local_fraction"] for row in rows
            ),
            "uncovered_fraction_mean": statistics.mean(
                row["uncovered_fraction"] for row in rows
            ),
            "mean_abs_offset": statistics.mean(row["mean_abs_offset"] for row in rows),
            "vanilla_wer_mean": vanilla_mean,
            "vanilla_wer_sd": vanilla_sd,
            "atdkd_wer_mean": atdkd_mean,
            "atdkd_wer_sd": atdkd_sd,
            "delta_wer_mean": delta_mean,
            "delta_wer_sd": delta_sd,
            "delta_wer_ci_low": ci_low,
            "delta_wer_ci_high": ci_high,
            "rerr_mean": rerr_mean,
            "rerr_sd": rerr_sd,
        })

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    md_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# Natural teacher-student mismatch stratification",
        "",
        "Bins were fixed before inspecting Vanilla/AT-DKD predictions, using the fraction of",
        "reference BPE occurrences with nonzero Teacher-No-KD offset whose No-KD peak is",
        "inside the teacher delta=6 occupancy support (gamma > 0.01). WER is no-LM beam-16",
        "mean +/- sample SD over three matched seeds. Delta WER is Vanilla minus AT-DKD;",
        "positive values favor AT-DKD. CI is a paired utterance-cluster bootstrap over the",
        "three-seed mean (2,000 replicates).",
        "",
        "| Recoverable-mismatch bin | Utts | Mean recoverable frac. | Mean uncovered frac. | Mean offset | Vanilla WER | AT-DKD WER | Delta WER [95% CI] | RERR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['label']} | {row['utterances']} | "
            f"{row['recoverable_fraction_mean']:.3f} | {row['uncovered_fraction_mean']:.3f} | "
            f"{row['mean_abs_offset']:.3f} | "
            f"{row['vanilla_wer_mean']:.2f} +/- {row['vanilla_wer_sd']:.2f} | "
            f"{row['atdkd_wer_mean']:.2f} +/- {row['atdkd_wer_sd']:.2f} | "
            f"{row['delta_wer_mean']:.2f} [{row['delta_wer_ci_low']:.2f}, {row['delta_wer_ci_high']:.2f}] | "
            f"{row['rerr_mean']:.2f} +/- {row['rerr_sd']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    x = np.arange(4)
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    axes[0].bar(x - width / 2, [r["vanilla_wer_mean"] for r in results], width,
                yerr=[r["vanilla_wer_sd"] for r in results], capsize=3,
                label="Vanilla KD", color="#d55e00")
    axes[0].bar(x + width / 2, [r["atdkd_wer_mean"] for r in results], width,
                yerr=[r["atdkd_wer_sd"] for r in results], capsize=3,
                label="AT-DKD", color="#009e73")
    axes[0].set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
    axes[0].set_xlabel("Recoverable local mismatch burden")
    axes[0].set_ylabel("WER (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    delta = np.array([r["delta_wer_mean"] for r in results])
    low = delta - np.array([r["delta_wer_ci_low"] for r in results])
    high = np.array([r["delta_wer_ci_high"] for r in results]) - delta
    axes[1].errorbar(x, delta, yerr=np.vstack([low, high]), marker="o", capsize=4,
                     color="#2f6fbb")
    axes[1].axhline(0, color="#555", linewidth=0.8)
    axes[1].set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
    axes[1].set_xlabel("Recoverable local mismatch burden")
    axes[1].set_ylabel("WER reduction (Vanilla - AT-DKD, %p)")
    axes[1].grid(alpha=0.25)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"wrote {csv_path}, {md_path}, PNG, and PDF")


if __name__ == "__main__":
    main()
