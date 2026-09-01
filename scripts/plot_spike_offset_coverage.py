#!/usr/bin/env python3
"""Plot teacher--student spike offsets covered by the delta=6 occupancy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "spike_offset_coverage_delta6"
EPS = 0.01

DATASETS = {
    "LibriSpeech test-clean": [ROOT / "analysis/spike_coverage/offsets.npz"],
    "CHiME-3 eval": [
        ROOT / "analysis/spike_coverage_chime3_eval_real_200/offsets.npz",
        ROOT / "analysis/spike_coverage_chime3_eval_simu_200/offsets.npz",
    ],
}

TEAL = "#168C8C"
LIGHT = "#E5E9EC"
INK = "#263238"
MUTED = "#66747A"
GRID = "#D7DEE1"


def load(paths):
    arrays = [np.load(path) for path in paths]
    offsets = np.concatenate([arr["no-KD_off"] for arr in arrays])
    coverage = np.concatenate([arr["no-KD_cov6"] for arr in arrays])
    return np.abs(offsets), coverage > EPS


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    bins = [
        ("0\nexact", lambda x: x == 0),
        ("1", lambda x: x == 1),
        ("2", lambda x: x == 2),
        ("≥3", lambda x: x >= 3),
    ]

    for ax, (name, paths) in zip(axes, DATASETS.items()):
        offsets, hit = load(paths)
        total = np.array([mask(offsets).mean() for _, mask in bins])
        covered = np.array([(mask(offsets) & hit).mean() for _, mask in bins])
        within = np.divide(covered, total, out=np.zeros_like(covered), where=total > 0)
        x = np.arange(len(bins))

        ax.bar(x, total * 100, width=0.68, color=LIGHT, edgecolor="white", linewidth=0.8)
        ax.bar(x, covered * 100, width=0.68, color=TEAL, edgecolor="white", linewidth=0.8)

        for xi, height, fraction in zip(x, total * 100, within * 100):
            ax.text(
                xi,
                height + 1.2,
                f"{fraction:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold" if xi == 1 else "normal",
                color=TEAL if fraction >= 15 else MUTED,
            )

        exact = (offsets == 0).mean() * 100
        overall = hit.mean() * 100
        ax.text(
            0.03,
            0.96,
            f"Exact: {exact:.1f}%   →   δ=6 covered: {overall:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=INK,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "lw": 0.8},
        )

        ax.set_title(name, pad=8, fontweight="bold")
        ax.set_xticks(x, [label for label, _ in bins])
        ax.set_xlabel(r"Absolute spike offset $|t_S-t_T|$ (frames)")
        ax.set_ylim(0, 54)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.tick_params(axis="y", length=3)

    axes[0].set_ylabel("Share of token occurrences (%)")
    axes[1].tick_params(axis="y", left=False)
    fig.legend(
        handles=[
            Patch(facecolor=TEAL, label="Covered by teacher occupancy (δ=6)"),
            Patch(facecolor=LIGHT, label="Not covered"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=2.1,
    )
    fig.suptitle(
        "Teacher occupancy absorbs local student spike misalignment",
        y=1.02,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.81, bottom=0.25, wspace=0.10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
