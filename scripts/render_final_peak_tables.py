#!/usr/bin/env python3
"""Render final occurrence-level peak statistics in a dark paper-table style."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "final_peak_analysis_tables"

BG = "#050505"
FG = "#F1F1F1"
MUTED = "#B8B8B8"
RULE = "#272727"
ACCENT = "#FFFFFF"


def draw_table(ax, columns, rows, widths, title, subtitle=None, bold_cols=()):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0, 1.03, title, color=FG, fontsize=15, fontweight="bold", va="bottom")
    if subtitle:
        ax.text(1, 1.03, subtitle, color=MUTED, fontsize=9, va="bottom", ha="right")

    xs = [0]
    for width in widths:
        xs.append(xs[-1] + width)

    header_y = 0.83
    row_h = 0.27
    for j, label in enumerate(columns):
        ax.text(
            xs[j] + 0.006,
            header_y,
            label,
            color=FG,
            fontsize=10.2,
            fontweight="bold",
            va="center",
            ha="left",
        )
    ax.plot([0, 1], [header_y - 0.10, header_y - 0.10], color=RULE, lw=1.0)

    for i, row in enumerate(rows):
        y = header_y - 0.23 - i * row_h
        for j, value in enumerate(row):
            ax.text(
                xs[j] + 0.006,
                y,
                value,
                color=ACCENT if j in bold_cols else FG,
                fontsize=10.3,
                fontweight="bold" if j in bold_cols else "normal",
                va="center",
                ha="left",
            )
        ax.plot([0, 1], [y - 0.13, y - 0.13], color=RULE, lw=0.8)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.2, 7.1), facecolor=BG)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42)

    ax1 = fig.add_subplot(grid[0])
    draw_table(
        ax1,
        [
            "Dataset",
            "Utts.",
            "Occurrences",
            r"Mean $|\Delta|$",
            "Exact match",
            "One-frame mismatch",
            r"$\geq$2-frame mismatch",
            r"$|\Delta|=1$ / mismatch",
        ],
        [
            [
                "LibriSpeech test-clean",
                "200",
                "8,237",
                "0.69",
                "3,577  (43.4%)",
                "3,818  (46.4%)",
                "842  (10.2%)",
                "81.9%",
            ],
            [
                "CHiME-3 eval",
                "400",
                "12,536",
                "1.82",
                "743  (5.9%)",
                "4,045  (32.3%)",
                "7,748  (61.8%)",
                "34.3%",
            ],
        ],
        [0.16, 0.06, 0.10, 0.09, 0.14, 0.16, 0.15, 0.14],
        "Teacher–No-KD student peak disagreement",
        "Matched reference BPE occurrences · seed 1",
    )

    ax2 = fig.add_subplot(grid[1])
    draw_table(
        ax2,
        [
            "Dataset",
            r"All · $\delta=0$",
            r"All · $\delta=6$",
            "All gain",
            r"$|\Delta|=1$ · $\delta=0$",
            r"$|\Delta|=1$ · $\delta=6$",
            r"$|\Delta|=1$ gain",
        ],
        [
            [
                "LibriSpeech test-clean",
                "48.4%  (3,989/8,237)",
                "73.0%  (6,016/8,237)",
                "+24.6 pp",
                "10.5%  (402/3,818)",
                "62.1%  (2,370/3,818)",
                "+51.5 pp",
            ],
            [
                "CHiME-3 eval",
                "11.7%  (1,468/12,536)",
                "28.4%  (3,563/12,536)",
                "+16.7 pp",
                "17.0%  (689/4,045)",
                "62.1%  (2,510/4,045)",
                "+45.0 pp",
            ],
        ],
        [0.17, 0.15, 0.16, 0.10, 0.16, 0.17, 0.09],
        "Student-peak coverage by teacher-derived occupancy",
        r"Hit if $\gamma_i^{(T,\delta)}(p_i^{(S)}) > 0.01$",
        bold_cols=(2, 5),
    )

    fig.text(
        0.025,
        0.015,
        "CHiME-3 combines eval-real (200 utterances) and eval-simu (200 utterances). "
        "The one-frame columns use only occurrences with |Δ|=1 as their denominator.",
        color=MUTED,
        fontsize=9,
        ha="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
