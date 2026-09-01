#!/usr/bin/env python3
"""Render the dataset-level peak-coverage results as a paper-ready table."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "dataset_peak_coverage_table"


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    columns = [
        "Dataset",
        "Teacher-derived support",
        "All peaks (%)",
        r"$|\Delta|=1$ peaks (%)",
    ]
    rows = [
        ["LibriSpeech\ntest-clean", r"Standard FB ($\delta=0$)", "48.4", "10.5"],
        ["", r"Blank-suppressed FB ($\delta=6$)", "73.0", "62.1"],
        ["CHiME-3 eval\n(real + simu)", r"Standard FB ($\delta=0$)", "11.7", "17.0"],
        ["", r"Blank-suppressed FB ($\delta=6$)", "28.4", "62.1"],
    ]

    fig, ax = plt.subplots(figsize=(8.3, 3.25))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        colWidths=[0.20, 0.42, 0.18, 0.20],
        bbox=[0.01, 0.19, 0.98, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    navy = "#243447"
    teal = "#147D7E"
    light_teal = "#E8F4F3"
    light_gray = "#F5F7F8"
    rule = "#CBD3D8"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(rule)
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(navy)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(0.16)
        else:
            cell.set_height(0.18)
            cell.set_facecolor(light_gray if r in (1, 3) else "white")
            if c in (0, 1):
                cell.get_text().set_ha("left")
                cell.PAD = 0.12

    for r in (2, 4):
        for c in range(4):
            table[(r, c)].set_facecolor(light_teal)
        table[(r, 1)].get_text().set_weight("bold")
        table[(r, 1)].get_text().set_color(teal)
        for c in (2, 3):
            table[(r, c)].get_text().set_weight("bold")
            table[(r, c)].get_text().set_color(teal)

    # Visually merge the repeated dataset cells without relying on row spans.
    for upper, lower in ((1, 2), (3, 4)):
        table[(upper, 0)].visible_edges = "LTR"
        table[(lower, 0)].visible_edges = "LBR"

    fig.text(
        0.01,
        0.95,
        "Dataset-level coverage of No-KD student peaks",
        fontsize=13,
        fontweight="bold",
        color=navy,
        ha="left",
    )
    fig.text(
        0.01,
        0.07,
        r"Coverage: teacher occupancy at the student peak $>0.01$. "
        r"The $|\Delta|=1$ column is restricted to one-frame mismatches.",
        fontsize=8.6,
        color="#52616B",
        ha="left",
    )
    fig.text(
        0.01,
        0.015,
        "LibriSpeech: test-clean, 200 utterances / 8,237 occurrences.  "
        "CHiME-3: eval real+simu, 400 utterances / 12,536 occurrences. Seed 1.",
        fontsize=8.3,
        color="#52616B",
        ha="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
