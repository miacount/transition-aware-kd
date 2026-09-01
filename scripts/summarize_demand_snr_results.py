#!/usr/bin/env python
"""Summarize and plot the LibriSpeech + DEMAND SNR stress experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


CONDITIONS = [
    ("clean", "Clean"),
    ("snr_p10", "+10"),
    ("snr_p0", "0"),
    ("snr_m5", "-5"),
    ("snr_m10", "-10"),
]


def beam_wer(log_path: Path, condition: str) -> float:
    text = log_path.read_text()
    block = text.split(f"[{condition}]", 1)[1]
    match = re.search(r"beam\s+WER=([0-9.]+)%", block)
    if match is None:
        raise ValueError(f"No beam WER for {condition} in {log_path}")
    return float(match.group(1))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("analysis/librispeech_demand_snr"))
    parser.add_argument("--output-prefix", type=Path, default=Path("analysis/librispeech_demand_snr/paper_summary"))
    args = parser.parse_args()

    rows = []
    for condition, label in CONDITIONS:
        vanilla = [beam_wer(args.input_dir / f"vanilla_s{s}_beam16.txt", condition) for s in (1, 2, 3)]
        atdkd = [beam_wer(args.input_dir / f"atdkd_s{s}_beam16.txt", condition) for s in (1, 2, 3)]
        vanilla_mean, vanilla_sd = mean_sd(vanilla)
        atdkd_mean, atdkd_sd = mean_sd(atdkd)
        paired_rerr = [100.0 * (v - a) / v for v, a in zip(vanilla, atdkd)]
        rerr_mean, rerr_sd = mean_sd(paired_rerr)
        spike = json.loads((args.input_dir / f"spike_{condition}_all.json").read_text())["no_kd"]["spike_off"]
        rows.append({
            "condition": condition,
            "snr_db": label,
            "n_utterances": 2703,
            "spike_offset_frames": spike,
            "vanilla_wer_mean": vanilla_mean,
            "vanilla_wer_sd": vanilla_sd,
            "atdkd_wer_mean": atdkd_mean,
            "atdkd_wer_sd": atdkd_sd,
            "paired_rerr_mean_pct": rerr_mean,
            "paired_rerr_sd_pct": rerr_sd,
        })

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# LibriSpeech dev-clean + DEMAND SNR stress test",
        "",
        "WER is no-LM beam-16, reported as mean ± sample SD over three matched seeds. "
        "Relative error-rate reduction (RERR) is computed per matched seed and then averaged. "
        "Spike offset is the transcript-constrained Teacher–No-KD mean absolute peak offset over all 2,703 utterances.",
        "",
        "| SNR (dB) | Spike offset (frames) | Vanilla KD WER (%) | AT-DKD WER (%) | Paired RERR (%) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['snr_db']} | {row['spike_offset_frames']:.2f} | "
            f"{row['vanilla_wer_mean']:.2f} ± {row['vanilla_wer_sd']:.2f} | "
            f"{row['atdkd_wer_mean']:.2f} ± {row['atdkd_wer_sd']:.2f} | "
            f"{row['paired_rerr_mean_pct']:.2f} ± {row['paired_rerr_sd_pct']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    labels = [row["snr_db"] for row in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(x, [row["spike_offset_frames"] for row in rows], marker="o", color="#2f6fbb")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Mean absolute spike offset (frames)")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, [row["vanilla_wer_mean"] for row in rows],
                     yerr=[row["vanilla_wer_sd"] for row in rows], marker="o", capsize=3,
                     label="Vanilla KD", color="#d55e00")
    axes[1].errorbar(x, [row["atdkd_wer_mean"] for row in rows],
                     yerr=[row["atdkd_wer_sd"] for row in rows], marker="s", capsize=3,
                     label="AT-DKD", color="#009e73")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("WER (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    for i, row in enumerate(rows):
        axes[1].annotate(f"{row['paired_rerr_mean_pct']:+.1f}%", (i, row["atdkd_wer_mean"]),
                         xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7)

    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"wrote {csv_path}, {md_path}, {args.output_prefix.with_suffix('.png')}, and PDF")


if __name__ == "__main__":
    main()
