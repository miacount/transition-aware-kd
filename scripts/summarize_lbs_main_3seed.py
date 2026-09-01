#!/usr/bin/env python3
"""Build per-seed and mean +/- sample-std tables for the LBS main suite."""
import argparse
import csv
import re
import statistics
from pathlib import Path


METHODS = (
    ("No KD", "nokd"),
    ("Vanilla frame KD", "vanilla"),
    ("Blank elimination KD", "kdbe"),
    ("Symmetric selection KD", "symmetric"),
    ("Guided CTC", "guided"),
    ("S-CTC + CTC fine-tune", "sctc_ft"),
    ("CR-CTC", "crctc"),
    ("Mass3 + NTDK, M=32 (ours)", "ours"),
)
SEEDS = (1, 2, 3)
SPLIT_RE = re.compile(r"^\[([^]]+)\]$")
WER_RE = re.compile(r"^(greedy|beam)\s+WER=([0-9]+(?:\.[0-9]+)?)%")
COLUMNS = (
    ("dev-clean G", "dev_clean", "greedy"),
    ("dev-other G", "dev_other", "greedy"),
    ("test-clean G", "test_clean", "greedy"),
    ("test-other G", "test_other", "greedy"),
    ("test-clean B", "test_clean", "beam"),
    ("test-other B", "test_other", "beam"),
)


def parse_log(path):
    result = {}
    split = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = SPLIT_RE.match(line)
        if match:
            split = match.group(1)
            continue
        match = WER_RE.match(line)
        if split and match:
            result[(split, match.group(1))] = float(match.group(2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    records = {}
    missing = []
    for _, method in METHODS:
        for seed in SEEDS:
            path = args.analysis_dir / f"beam{args.beam_width}_main3seed_lbs_{method}_s{seed}.txt"
            if not path.is_file():
                missing.append(str(path))
                continue
            records[(method, seed)] = parse_log(path)
    if missing:
        raise SystemExit("missing logs:\n  " + "\n  ".join(missing))

    csv_path = args.analysis_dir / f"lbs_main_3seed_m32_beam{args.beam_width}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seed", "split", "decoder", "wer_percent"])
        for label, method in METHODS:
            for seed in SEEDS:
                for _, split, decoder in COLUMNS:
                    key = (split, decoder)
                    if key not in records[(method, seed)]:
                        raise SystemExit(f"{method} seed {seed} lacks {key}")
                    writer.writerow([label, seed, split, decoder, f"{records[(method, seed)][key]:.4f}"])

    md_path = args.analysis_dir / f"lbs_main_3seed_m32_beam{args.beam_width}.md"
    header = "| Method | " + " | ".join(column[0] for column in COLUMNS) + " |"
    separator = "|---|" + "---:|" * len(COLUMNS)
    lines = [
        "# LibriSpeech main results: three matched seeds",
        "",
        "WER (%), mean +/- sample standard deviation over seeds 1/2/3. "
        f"G=greedy and B=beam-{args.beam_width} without an external LM.",
        "",
        header,
        separator,
    ]
    for label, method in METHODS:
        cells = []
        for _, split, decoder in COLUMNS:
            values = [records[(method, seed)][(split, decoder)] for seed in SEEDS]
            cells.append(f"{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += ["", "## Per-seed beam WER", "", "| Method | Seed | test-clean | test-other |", "|---|---:|---:|---:|"]
    for label, method in METHODS:
        for seed in SEEDS:
            result = records[(method, seed)]
            lines.append(
                f"| {label} | {seed} | {result[('test_clean', 'beam')]:.4f} | "
                f"{result[('test_other', 'beam')]:.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
