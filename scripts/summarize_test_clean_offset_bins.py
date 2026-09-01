#!/usr/bin/env python
"""Paper table stratified by robust utterance-level Teacher-NoKD offset."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from summarize_natural_mismatch import edit_distance


EPS = 0.01


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test_clean", "test_other"),
                        default="test_clean")
    parser.add_argument("--offset-stat", choices=("median", "robust_mean"),
                        default="median")
    args = parser.parse_args()
    root = Path(f"analysis/natural_mismatch_{args.split}")
    coverage_root = Path(f"analysis/natural_mismatch_{args.split}_delta9")
    predictions = root / "predictions"

    mismatch_rows = read_jsonl(coverage_root / "per_utterance_no_kd.jsonl")
    arrays = np.load(coverage_root / "offsets.npz")
    offsets = arrays["no_kd_off"]
    cov6 = arrays["no_kd_cov6"]
    cov9 = arrays["no_kd_cov9"]
    assert sum(row["n_tokens"] for row in mismatch_rows) == len(offsets)

    token_slices = {}
    removed_outliers = 0
    cursor = 0
    for row in mismatch_rows:
        stop = cursor + row["n_tokens"]
        utt = row["utterance_id"]
        token_slices[utt] = slice(cursor, stop)
        absolute = np.abs(offsets[cursor:stop])
        median = float(np.median(absolute))
        robust_values = absolute
        if len(absolute) >= 3:
            ordered = np.sort(absolute)
            if ordered[-1] - median >= 3 and ordered[-1] - ordered[-2] >= 2:
                robust_values = ordered[:-1]
                removed_outliers += 1
        row["median_abs_offset"] = median
        row["robust_mean_abs_offset"] = float(robust_values.mean())
        row["group_offset"] = row[
            "median_abs_offset" if args.offset_stat == "median" else "robust_mean_abs_offset"
        ]
        cursor = stop

    groups = {
        "0 frames": [row["utterance_id"] for row in mismatch_rows
                     if row["group_offset"] < 0.5],
        "1 frame": [row["utterance_id"] for row in mismatch_rows
                    if 0.5 <= row["group_offset"] < 1.5],
        ">=2 frames": [row["utterance_id"] for row in mismatch_rows
                       if row["group_offset"] >= 1.5],
    }
    assigned = sum(len(ids) for ids in groups.values())
    assert assigned == len(mismatch_rows)

    counts = {"teacher": {}}
    teacher_rows = read_jsonl(predictions / f"teacher_{args.split}.jsonl")
    counts["teacher"] = {
        row["utterance_id"]: edit_distance(row["reference"], row["beam"])
        for row in teacher_rows
    }
    for method in ("vanilla", "atdkd"):
        counts[method] = {}
        seeds = (1, 2, 3) if args.split == "test_clean" else (1,)
        for seed in seeds:
            if method == "atdkd" and args.split == "test_other":
                filename = f"atdkd_d6_s{seed}_{args.split}.jsonl"
            else:
                filename = f"{method}_s{seed}_{args.split}.jsonl"
            rows = read_jsonl(predictions / filename)
            counts[method][seed] = {
                row["utterance_id"]: edit_distance(row["reference"], row["beam"])
                for row in rows
            }
    d9_rows = read_jsonl(predictions / f"atdkd_d9_s1_{args.split}.jsonl")
    counts["atdkd_d9_s1"] = {
        row["utterance_id"]: edit_distance(row["reference"], row["beam"])
        for row in d9_rows
    }
    expected = set(token_slices)
    assert set(counts["teacher"]) == expected
    for method in ("vanilla", "atdkd"):
        for seed in counts[method]:
            assert set(counts[method][seed]) == expected
    assert set(counts["atdkd_d9_s1"]) == expected

    results = []
    table_groups = list(groups.items()) + [("Overall", list(expected))]
    for label, ids in table_groups:
        token_indices = np.concatenate([
            np.arange(token_slices[utt].start, token_slices[utt].stop) for utt in ids
        ])
        teacher_edits = sum(counts["teacher"][utt][0] for utt in ids)
        words = sum(counts["teacher"][utt][1] for utt in ids)
        teacher_wer = 100.0 * teacher_edits / words
        vanilla_wers, atdkd_wers, rerrs = [], [], []
        for seed in counts["vanilla"]:
            v_edits = sum(counts["vanilla"][seed][utt][0] for utt in ids)
            a_edits = sum(counts["atdkd"][seed][utt][0] for utt in ids)
            v_words = sum(counts["vanilla"][seed][utt][1] for utt in ids)
            assert v_words == words
            v_wer = 100.0 * v_edits / words
            a_wer = 100.0 * a_edits / words
            vanilla_wers.append(v_wer)
            atdkd_wers.append(a_wer)
            rerrs.append(100.0 * (v_wer - a_wer) / v_wer)
        vm = statistics.mean(vanilla_wers)
        vs = statistics.stdev(vanilla_wers) if len(vanilla_wers) > 1 else 0.0
        am = statistics.mean(atdkd_wers)
        ass = statistics.stdev(atdkd_wers) if len(atdkd_wers) > 1 else 0.0
        rm = statistics.mean(rerrs)
        rs = statistics.stdev(rerrs) if len(rerrs) > 1 else 0.0
        vanilla_s1 = vanilla_wers[0]
        atdkd_d6_s1 = atdkd_wers[0]
        d9_edits = sum(counts["atdkd_d9_s1"][utt][0] for utt in ids)
        atdkd_d9_s1 = 100.0 * d9_edits / words
        results.append({
            "offset_bin": label,
            "utterances": len(ids),
            "utterance_share": len(ids) / len(mismatch_rows),
            "tokens": len(token_indices),
            "coverage_delta6": float((cov6[token_indices] > EPS).mean()),
            "coverage_delta9": float((cov9[token_indices] > EPS).mean()),
            "teacher_wer": teacher_wer,
            "vanilla_s1_wer": vanilla_s1,
            "atdkd_d6_s1_wer": atdkd_d6_s1,
            "atdkd_d6_s1_rerr": 100.0 * (vanilla_s1 - atdkd_d6_s1) / vanilla_s1,
            "atdkd_d9_s1_wer": atdkd_d9_s1,
            "atdkd_d9_s1_rerr": 100.0 * (vanilla_s1 - atdkd_d9_s1) / vanilla_s1,
            "vanilla_wer_mean": vm,
            "vanilla_wer_sd": vs,
            "atdkd_wer_mean": am,
            "atdkd_wer_sd": ass,
            "relative_werr_mean": rm,
            "relative_werr_sd": rs,
        })

    stat_name = "median" if args.offset_stat == "median" else "robust_mean"
    output = root / f"{args.split}_{stat_name}_offset_bins"
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    offset_title = "Median" if args.offset_stat == "median" else "Mean"
    if args.offset_stat == "robust_mean":
        setup_text = (
            "Coverage is the token-level fraction whose NoKD peak has teacher occupancy mass > 0.01. "
            "All WERs use matched seed 1 and no-LM beam-16; RERR is relative to Vanilla KD in the same row."
        )
    else:
        setup_text = (
            "Coverage is the token-level fraction whose NoKD peak has teacher occupancy mass > 0.01. "
            "All WERs use no-LM beam-16. The delta=6 versus delta=9 comparison uses matched seed 1; "
            "RERR is relative to the seed-1 Vanilla KD WER in the same row."
        )
    lines = [
        f"# {args.split.replace('_', '-')} by {offset_title.lower()} Teacher-NoKD peak offset",
        "",
        setup_text,
        f"{offset_title} bins use [0, 0.5), [0.5, 1.5), and [1.5, infinity) frame boundaries.",
        "",
    ]
    if args.offset_stat == "robust_mean":
        lines += [
            f"An isolated maximum was removed in {removed_outliers} utterances when it was at least "
            "3 frames above the median and at least 2 frames above the second-largest offset.",
            "",
            "| Mean peak offset* | Utt. share | Cov.@d=6 | Teacher | Vanilla KD | AT-DKD d=6 | RERR d=6 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            "| Median peak offset | Utt. share | Coverage@d=6 | Coverage@d=9 | Teacher WER | Vanilla KD (s1) | AT-DKD d=6 (s1) | RERR d=6 | AT-DKD d=9 (s1) | RERR d=9 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in results:
        if args.offset_stat == "robust_mean":
            lines.append(
                f"| {row['offset_bin']} | {100*row['utterance_share']:.1f}% "
                f"({row['utterances']}) | {100*row['coverage_delta6']:.1f}% | "
                f"{row['teacher_wer']:.2f} | {row['vanilla_s1_wer']:.2f} | "
                f"{row['atdkd_d6_s1_wer']:.2f} | {row['atdkd_d6_s1_rerr']:.2f}% |"
            )
        else:
            lines.append(
                f"| {row['offset_bin']} | {100*row['utterance_share']:.1f}% "
                f"({row['utterances']}) | {100*row['coverage_delta6']:.1f}% | "
                f"{100*row['coverage_delta9']:.1f}% | {row['teacher_wer']:.2f} | "
                f"{row['vanilla_s1_wer']:.2f} | {row['atdkd_d6_s1_wer']:.2f} | "
                f"{row['atdkd_d6_s1_rerr']:.2f}% | {row['atdkd_d9_s1_wer']:.2f} | "
                f"{row['atdkd_d9_s1_rerr']:.2f}% |"
            )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")

    if args.offset_stat == "robust_mean":
        latex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
                 "Mean peak offset$^{*}$ & Utt. share & Cov.@$\\delta$=6 & Teacher & Vanilla & AT-DKD $\\delta$=6 & RERR \\\\",
                 "\\midrule"]
    else:
        latex = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
                 "Median peak offset & Utt. share & Cov.@$\\delta$=6 & Cov.@$\\delta$=9 & Teacher & Vanilla & AT-DKD $\\delta$=6 & RERR & AT-DKD $\\delta$=9 & RERR \\\\",
                 "\\midrule"]
    for row in results:
        label = row['offset_bin'].replace('>=', '$\\geq$')
        if args.offset_stat == "robust_mean":
            latex.append(
                f"{label} & {100*row['utterance_share']:.1f}\\% & "
                f"{100*row['coverage_delta6']:.1f}\\% & {row['teacher_wer']:.2f} & "
                f"{row['vanilla_s1_wer']:.2f} & {row['atdkd_d6_s1_wer']:.2f} & "
                f"{row['atdkd_d6_s1_rerr']:.2f}\\% \\\\"
            )
        else:
            latex.append(
                f"{label} & {100*row['utterance_share']:.1f}\\% & {100*row['coverage_delta6']:.1f}\\% & "
                f"{100*row['coverage_delta9']:.1f}\\% & {row['teacher_wer']:.2f} & "
                f"{row['vanilla_s1_wer']:.2f} & {row['atdkd_d6_s1_wer']:.2f} & "
                f"{row['atdkd_d6_s1_rerr']:.2f}\\% & {row['atdkd_d9_s1_wer']:.2f} & "
                f"{row['atdkd_d9_s1_rerr']:.2f}\\% \\\\"
            )
    latex += ["\\bottomrule", "\\end{tabular}"]
    output.with_suffix(".tex").write_text("\n".join(latex) + "\n")
    print(f"wrote {output}.csv/.md/.tex")


if __name__ == "__main__":
    main()
