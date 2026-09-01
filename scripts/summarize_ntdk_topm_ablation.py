#!/usr/bin/env python3
"""Summarize beam WER logs for the LibriSpeech NTDK top-M ablation."""
import argparse
import re
from pathlib import Path


SECTION = re.compile(r"^\[([^]]+)\]$")
WER = re.compile(r"^beam\s+WER=([0-9]+(?:\.[0-9]+)?)%")


def parse_log(path):
    values = {}
    split = None
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            match = SECTION.match(line)
            if match:
                split = match.group(1)
                continue
            match = WER.match(line)
            if split and match:
                values[split] = float(match.group(1))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    top_ms = (1, 2, 3, 4, 8, 16, 32)
    logs = {
        top_m: (
            args.analysis_dir / f"beam{args.beam_width}_paper_lbs_mass25_ntdk8.txt"
            if top_m == 32 else
            args.analysis_dir / f"beam{args.beam_width}_ablation_lbs_ntdk_topm{top_m}.txt"
        )
        for top_m in top_ms
    }
    missing = [str(path) for path in logs.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing evaluation logs: " + ", ".join(missing))
    results = {top_m: parse_log(path) for top_m, path in logs.items()}
    splits = ("dev_clean", "dev_other", "test_clean", "test_other")
    for top_m, values in results.items():
        absent = [split for split in splits if split not in values]
        if absent:
            raise SystemExit(f"{logs[top_m]} lacks {absent}")

    tsv = args.analysis_dir / f"ntdk_topm_ablation_beam{args.beam_width}.tsv"
    md = args.analysis_dir / f"ntdk_topm_ablation_beam{args.beam_width}.md"
    lines = ["top_m\t" + "\t".join(splits)]
    table = ["| M | " + " | ".join(splits) + " |", "|---:|---:|---:|---:|---:|"]
    for top_m in top_ms:
        values = results[top_m]
        formatted = [f"{values[split]:.4f}" for split in splits]
        lines.append(str(top_m) + "\t" + "\t".join(formatted))
        table.append(f"| {top_m} | " + " | ".join(formatted) + " |")
    tsv.write_text("\n".join(lines) + "\n")
    md.write_text("# NTDK top-M+tail ablation (beam WER, %)\n\n" + "\n".join(table) + "\n")
    print("\n".join(table))
    print(f"[saved] {tsv}")
    print(f"[saved] {md}")


if __name__ == "__main__":
    main()
