#!/usr/bin/env python3
"""Summarize the LibriSpeech + CHIME-3 three-seed paper main table."""
import argparse
import csv
import re
import statistics
from pathlib import Path


METHODS = (
    ("No-KD", "nokd"),
    ("Vanilla frame KD", "vanilla"),
    ("Symmetric Selection", "symmetric"),
    ("Guided CTC", "guided"),
    ("S-CTC + CTC FT", "sctc_ft"),
    ("FPKD", "fpkd"),
    ("CARL", "carl"),
    ("CR-CTC", "crctc"),
    ("AT-DKD (Ours, M=32)", "ours"),
)
SEEDS = (1, 2, 3)
DATASETS = {
    "lbs": ("dev_clean", "dev_other", "test_clean", "test_other"),
    "chime3": ("chime3_dev_real", "chime3_dev_simu", "chime3_eval_real", "chime3_eval_simu"),
}
HEADERS = ("LS Dev Clean", "LS Dev Other", "LS Test Clean", "LS Test Other",
           "CH Dev Real", "CH Dev Sim.", "CH Eval Real", "CH Eval Sim.")
SPLIT_RE = re.compile(r"^\[([^]]+)\]$")
WER_RE = re.compile(r"^beam\s+WER=([0-9]+(?:\.[0-9]+)?)%")


def parse_beam(path):
    result, split = {}, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = SPLIT_RE.match(line)
        if match:
            split = match.group(1)
            continue
        match = WER_RE.match(line)
        if split and match:
            result[split] = float(match.group(1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    records, missing = {}, []
    for dataset in DATASETS:
        for _, method in METHODS:
            for seed in SEEDS:
                path = args.analysis_dir / f"beam{args.beam_width}_main3seed_{dataset}_{method}_s{seed}.txt"
                if not path.is_file():
                    missing.append(str(path))
                else:
                    records[(dataset, method, seed)] = parse_beam(path)
    if missing:
        raise SystemExit("missing logs:\n  " + "\n  ".join(missing))

    teacher_path = args.analysis_dir / "beam16_teacher_lbs_chime3.txt"
    if not teacher_path.is_file():
        raise SystemExit(f"missing teacher reference: {teacher_path}")
    teacher = parse_beam(teacher_path)
    all_splits = DATASETS["lbs"] + DATASETS["chime3"]
    absent = [split for split in all_splits if split not in teacher]
    if absent:
        raise SystemExit(f"teacher log lacks {absent}")

    csv_path = args.analysis_dir / f"main_table_3seed_m32_beam{args.beam_width}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seed", "dataset", "split", "beam_wer_percent"])
        for label, method in METHODS:
            for dataset, splits in DATASETS.items():
                for seed in SEEDS:
                    for split in splits:
                        value = records[(dataset, method, seed)].get(split)
                        if value is None:
                            raise SystemExit(f"{dataset}/{method}/seed{seed} lacks {split}")
                        writer.writerow([label, seed, dataset, split, f"{value:.4f}"])

    md_path = args.analysis_dir / f"main_table_3seed_m32_beam{args.beam_width}.md"
    lines = [
        "# Main table: three matched student seeds",
        "",
        f"No-LM beam-{args.beam_width} WER (%). Student rows are mean +/- sample standard "
        "deviation over seeds 1/2/3; Teacher is a fixed reference.",
        "",
        "| Method | " + " | ".join(HEADERS) + " |",
        "|---|" + "---:|" * len(HEADERS),
        "| Teacher | " + " | ".join(f"{teacher[s]:.2f}" for s in all_splits) + " |",
    ]
    for label, method in METHODS:
        cells = []
        for dataset, splits in DATASETS.items():
            for split in splits:
                values = [records[(dataset, method, seed)][split] for seed in SEEDS]
                cells.append(f"{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")

    lines += ["", "## Per-seed beam WER", ""]
    for dataset, splits in DATASETS.items():
        lines += [f"### {dataset}", "", "| Method | Seed | " + " | ".join(splits) + " |",
                  "|---|---:|" + "---:|" * len(splits)]
        for label, method in METHODS:
            for seed in SEEDS:
                values = records[(dataset, method, seed)]
                lines.append(f"| {label} | {seed} | " + " | ".join(f"{values[s]:.4f}" for s in splits) + " |")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
