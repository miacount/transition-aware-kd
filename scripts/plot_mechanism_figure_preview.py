#!/usr/bin/env python3
"""Render paper-layout previews from existing pilot diagnostics."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#3568B8"
ORANGE = "#E1812C"
GREEN = "#3A923A"
RED = "#C44E52"
GRAY = "#8A8A8A"


def finish(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def misalignment():
    data = json.loads((ROOT / "analysis/spike_coverage/summary.json").read_text())
    student = data["students"]["no-KD"]
    strata = student["by_stratum"]
    labels = ["0", "1", "2", "3", ">=4"]
    shares = [100 * strata[k]["share"] for k in labels]
    deltas = data["deltas"]
    overall = [100 * strata["all"][f"delta={d:g}"]["hit"] for d in deltas]
    by_offset = {
        d: [100 * strata[k][f"delta={d:g}"]["hit"] for k in labels]
        for d in (0.0, 6.0, 12.0)
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.25))
    ax = axes[0]
    bars = ax.bar(labels, shares, color=[GRAY, ORANGE, BLUE, BLUE, BLUE])
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylim(0, 52)
    ax.set_xlabel(r"Absolute spike offset $|\Delta t|$ (frames)")
    ax.set_ylabel("Token occurrences (%)")
    ax.set_title("(a) Misalignment is mostly local")

    ax = axes[1]
    ax.plot(deltas, overall, marker="o", lw=2.2, color=BLUE)
    ax.scatter([0, 6], [overall[0], overall[2]], s=55, color=[GRAY, ORANGE], zorder=3)
    ax.annotate(f"{overall[0]:.1f}%", (0, overall[0]), xytext=(5, -15), textcoords="offset points")
    ax.annotate(f"{overall[2]:.1f}%", (6, overall[2]), xytext=(3, 7), textcoords="offset points")
    ax.set_xticks(deltas)
    ax.set_ylim(40, 96)
    ax.set_xlabel(r"Blank penalty $\delta$")
    ax.set_ylabel("Student-spike coverage (%)")
    ax.set_title("(b) Wider FB support recovers spikes")

    ax = axes[2]
    x = np.arange(len(labels)); width = .24
    for j, (d, color) in enumerate([(0.0, GRAY), (6.0, ORANGE), (12.0, BLUE)]):
        ax.bar(x + (j - 1) * width, by_offset[d], width, label=rf"$\delta={int(d)}$", color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_xlabel(r"Absolute spike offset $|\Delta t|$")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("(c) Recovery by mismatch size")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")

    fig.suptitle("Occurrence support makes CTC distillation alignment-tolerant", fontsize=13, y=1.04)
    fig.text(.995, .005, "Layout preview · existing LS test-clean pilot diagnostics", ha="right", fontsize=7, color="#666")
    fig.tight_layout()
    finish(fig, OUT / "preview_misalignment_recovery.png")


def dark_knowledge():
    data = json.loads((ROOT / "analysis/standardized_dark_other.json").read_text())
    raw = data["sources"]["raw"]
    no_kd = raw["students"]["no_kd"]
    ours = raw["students"]["span_kd"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.25))
    ax = axes[0]
    vals = [raw["neff_median"], raw["nt_neff_median"]]
    bars = ax.bar(["Full pooled\nposterior", "Conditional NT\n(remove blank+GT)"], vals,
                  color=[GRAY, ORANGE], width=.62)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=9, padding=3)
    ax.set_ylabel("Median effective classes")
    ax.set_ylim(0, 6.8)
    ax.set_title("(a) Conditioning reveals alternatives")

    ax = axes[1]
    mass = [100 * raw["top32_mass_median"], 100 * raw["top64_mass_median"]]
    bars = ax.bar(["Top-32 + tail", "Top-64 + tail"], mass, color=[BLUE, GREEN], width=.58)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=3)
    ax.set_ylim(90, 101)
    ax.set_ylabel("Median conditional mass retained")
    ax.set_title("(b) Compact and stable")
    ax.text(.5, .08,
            f"30 dB stability\nJaccard={raw['stable_jaccard32_median']:.3f}  ·  JS={raw['stable_js_median']:.4f}",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#BBBBBB"))

    ax = axes[2]
    metrics = ["MRR", "Recall@8", "Recall@32"]
    a = [100 * no_kd["mrr"], no_kd["recall_at_8_pct"], no_kd["recall_at_32_pct"]]
    b = [100 * ours["mrr"], ours["recall_at_8_pct"], ours["recall_at_32_pct"]]
    x = np.arange(3); width = .34
    ax.bar(x - width / 2, a, width, label="No KD", color=GRAY)
    ax.bar(x + width / 2, b, width, label="Ours pilot", color=ORANGE)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Teacher-consistent student errors (%)")
    ax.set_title("(c) Student reflects teacher confusions")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Dark knowledge is hidden by the CTC peak, not absent", fontsize=13, y=1.04)
    fig.text(.995, .005, "Layout preview · existing LS test-other pilot diagnostics", ha="right", fontsize=7, color="#666")
    fig.tight_layout()
    finish(fig, OUT / "preview_dark_knowledge.png")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": .18,
    })
    misalignment()
    dark_knowledge()
