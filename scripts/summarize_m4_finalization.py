#!/usr/bin/env python3
"""Summarize M=4 finalization against frozen M=32 results and controls."""

import argparse
import re
import statistics
from pathlib import Path


SPLITS = {
    "lbs": ("dev_clean", "dev_other", "test_clean", "test_other"),
    "chime3": ("chime3_dev_real", "chime3_dev_simu", "chime3_eval_real", "chime3_eval_simu"),
}
HEADERS = ("LS Dev Clean", "LS Dev Other", "LS Test Clean", "LS Test Other",
           "CH Dev Real", "CH Dev Sim.", "CH Eval Real", "CH Eval Sim.")
SECTION = re.compile(r"^\[([^]]+)\]$")
WER = re.compile(r"^beam\s+WER=([0-9]+(?:\.[0-9]+)?)%")


def parse(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values, split = {}, None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        match = SECTION.match(line)
        if match:
            split = match.group(1)
            continue
        match = WER.match(line)
        if split and match:
            values[split] = float(match.group(1))
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam-width", type=int, default=16)
    ap.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    a, beam = args.analysis_dir, args.beam_width

    records = {}
    for m in (4, 32):
        for dataset, splits in SPLITS.items():
            for seed in (1, 2, 3):
                path = (a / f"beam{beam}_m4final_{dataset}_full_d6_s{seed}.txt"
                        if m == 4 else
                        a / f"beam{beam}_main3seed_{dataset}_ours_s{seed}.txt")
                values = parse(path)
                absent = [s for s in splits if s not in values]
                if absent:
                    raise RuntimeError(f"{path} lacks {absent}")
                records[(m, dataset, seed)] = values

    lines = [
        "# M=4 finalization report", "",
        f"No-LM beam-{beam} WER (%); mean +/- sample standard deviation over matched seeds 1/2/3.", "",
        "## Matched three-seed comparison", "",
        "| Setting | " + " | ".join(HEADERS) + " |",
        "|---|" + "---:|" * len(HEADERS),
    ]
    means = {}
    for m in (4, 32):
        cells = []
        flat_index = 0
        for dataset, splits in SPLITS.items():
            for split in splits:
                vals = [records[(m, dataset, s)][split] for s in (1, 2, 3)]
                means[(m, flat_index)] = statistics.mean(vals)
                cells.append(f"{statistics.mean(vals):.2f} +/- {statistics.stdev(vals):.2f}")
                flat_index += 1
        lines.append(f"| AT-DKD, M={m} | " + " | ".join(cells) + " |")
    lines += ["", "### M=4 minus M=32 mean WER", "",
              "| " + " | ".join(HEADERS) + " |",
              "|" + "---:|" * len(HEADERS),
              "| " + " | ".join(f"{means[(4, i)] - means[(32, i)]:+.2f}" for i in range(8)) + " |", ""]

    fixed = {
        "No KD": {
            "lbs": a / f"beam{beam}_main3seed_lbs_nokd_s1.txt",
            "chime3": a / f"beam{beam}_main3seed_chime3_nokd_s1.txt",
        },
        "Mass3 only (delta=6)": {
            "lbs": a / f"beam{beam}_ablation_lbs_mass3_only.txt",
            "chime3": a / f"beam{beam}_ablation_chime3_mass3_only.txt",
        },
        "NTDK only (M=4, delta=6)": {
            "lbs": a / f"beam{beam}_m4final_lbs_ntdk_only_s1.txt",
            "chime3": a / f"beam{beam}_m4final_chime3_ntdk_only_s1.txt",
        },
        "Full (M=4, delta=0)": {
            "lbs": a / f"beam{beam}_m4final_lbs_full_d0_s1.txt",
            "chime3": a / f"beam{beam}_m4final_chime3_full_d0_s1.txt",
        },
        "Full (M=4, delta=6)": {
            "lbs": a / f"beam{beam}_m4final_lbs_full_d6_s1.txt",
            "chime3": a / f"beam{beam}_m4final_chime3_full_d6_s1.txt",
        },
        "Full (M=4, delta=9)": {
            "lbs": a / f"beam{beam}_m4final_lbs_full_d9_s1.txt",
            "chime3": a / f"beam{beam}_m4final_chime3_full_d9_s1.txt",
        },
    }
    lines += ["## M=4 component and delta ablation (seed 1)", "",
              "| Variant | LS Test Clean | LS Test Other | CH Eval Real | CH Eval Sim. |",
              "|---|---:|---:|---:|---:|"]
    for label, paths in fixed.items():
        lbs, ch = parse(paths["lbs"]), parse(paths["chime3"])
        lines.append(
            f"| {label} | {lbs['test_clean']:.2f} | {lbs['test_other']:.2f} | "
            f"{ch['chime3_eval_real']:.2f} | {ch['chime3_eval_simu']:.2f} |")

    correct = parse(a / f"beam{beam}_m4final_lbs_full_d6_s1.txt")
    perm = parse(a / f"beam{beam}_m4final_lbs_full_d6_permuted_s1.txt")
    lines += ["", "## M=4 semantic control (LibriSpeech, seed 1)", "",
              "| Variant | Test Clean | Test Other |", "|---|---:|---:|",
              f"| Correct NTDK | {correct['test_clean']:.2f} | {correct['test_other']:.2f} |",
              f"| Permuted NTDK probabilities | {perm['test_clean']:.2f} | {perm['test_other']:.2f} |",
              f"| Permuted - correct | {perm['test_clean']-correct['test_clean']:+.2f} | "
              f"{perm['test_other']-correct['test_other']:+.2f} |", ""]

    out = a / f"m4_finalization_beam{beam}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
