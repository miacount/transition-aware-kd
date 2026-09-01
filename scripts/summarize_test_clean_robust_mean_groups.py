#!/usr/bin/env python
"""Make two-bin and tertile test-clean tables from robust mean spike offset."""

import csv
import json
from pathlib import Path

import numpy as np

from summarize_natural_mismatch import edit_distance


ROOT = Path("analysis/natural_mismatch_test_clean")
COVERAGE_ROOT = Path("analysis/natural_mismatch_test_clean_delta9")
PREDICTIONS = ROOT / "predictions"
EPS = 0.01
Q1 = 7.0 / 12.0
Q2 = 19.0 / 27.0


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def counts(path):
    return {
        row["utterance_id"]: edit_distance(row["reference"], row["beam"])
        for row in read_jsonl(path)
    }


def main():
    rows = read_jsonl(COVERAGE_ROOT / "per_utterance_no_kd.jsonl")
    arrays = np.load(COVERAGE_ROOT / "offsets.npz")
    offsets = np.abs(arrays["no_kd_off"])
    cov6 = arrays["no_kd_cov6"]

    slices, robust_mean = {}, {}
    cursor = removed = 0
    for row in rows:
        stop = cursor + row["n_tokens"]
        values = offsets[cursor:stop]
        median = float(np.median(values))
        kept = values
        if len(values) >= 3:
            ordered = np.sort(values)
            if ordered[-1] - median >= 3 and ordered[-1] - ordered[-2] >= 2:
                kept = ordered[:-1]
                removed += 1
        utt = row["utterance_id"]
        slices[utt] = slice(cursor, stop)
        robust_mean[utt] = float(kept.mean())
        cursor = stop
    assert cursor == len(offsets)

    teacher = counts(PREDICTIONS / "teacher_test_clean.jsonl")
    vanilla = counts(PREDICTIONS / "vanilla_s1_test_clean.jsonl")
    atdkd = counts(PREDICTIONS / "atdkd_s1_test_clean.jsonl")
    atdkd_d9 = counts(PREDICTIONS / "atdkd_d9_s1_test_clean.jsonl")
    assert set(robust_mean) == set(teacher) == set(vanilla) == set(atdkd) == set(atdkd_d9)

    schemes = {
        "Two groups": (
            ("Low", "<0.500", lambda x: x < 0.5),
            ("Mismatch", ">=0.500", lambda x: x >= 0.5),
        ),
        "Tertiles": (
            ("Low", f"<{Q1:.3f}", lambda x: x < Q1),
            ("Medium", f"{Q1:.3f}-<{Q2:.3f}", lambda x: Q1 <= x < Q2),
            ("High", f">={Q2:.3f}", lambda x: x >= Q2),
        ),
    }

    results = []
    for scheme, definitions in schemes.items():
        for label, frame_range, predicate in definitions:
            ids = [utt for utt, value in robust_mean.items() if predicate(value)]
            token_index = np.concatenate([
                np.arange(slices[utt].start, slices[utt].stop) for utt in ids
            ])
            words = sum(vanilla[utt][1] for utt in ids)
            teacher_wer = 100.0 * sum(teacher[utt][0] for utt in ids) / words
            vanilla_wer = 100.0 * sum(vanilla[utt][0] for utt in ids) / words
            atdkd_wer = 100.0 * sum(atdkd[utt][0] for utt in ids) / words
            atdkd_d9_wer = 100.0 * sum(atdkd_d9[utt][0] for utt in ids) / words
            results.append({
                "scheme": scheme,
                "group": label,
                "mean_offset_range": frame_range,
                "utterances": len(ids),
                "utterance_share": len(ids) / len(rows),
                "mean_robust_offset": float(np.mean([robust_mean[utt] for utt in ids])),
                "coverage_delta6": float((cov6[token_index] > EPS).mean()),
                "teacher_wer": teacher_wer,
                "vanilla_wer": vanilla_wer,
                "atdkd_delta6_wer": atdkd_wer,
                "rerr_delta6": 100.0 * (vanilla_wer - atdkd_wer) / vanilla_wer,
                "atdkd_delta9_wer": atdkd_d9_wer,
                "rerr_delta9": 100.0 * (vanilla_wer - atdkd_d9_wer) / vanilla_wer,
                "delta9_gain_over_delta6": atdkd_wer - atdkd_d9_wer,
            })

    output = ROOT / "test_clean_robust_mean_groupings"
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# Test-clean robust mean offset groupings",
        "",
        "An isolated maximum is removed only when it is at least 3 frames above the median "
        "and at least 2 frames above the second-largest offset. This affected "
        f"{removed} of {len(rows)} utterances. All WERs are matched seed 1, no-LM beam-16.",
    ]
    for scheme in schemes:
        lines += [
            "",
            f"## {scheme}",
            "",
            "| Mean peak offset group | Range (frames) | Utt. share | Mean offset | Cov.@d=6 | Teacher | Vanilla KD | AT-DKD d=6 | RERR d=6 | AT-DKD d=9 | RERR d=9 | D9 gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in results:
            if row["scheme"] != scheme:
                continue
            lines.append(
                f"| {row['group']} | {row['mean_offset_range']} | "
                f"{100*row['utterance_share']:.1f}% ({row['utterances']}) | "
                f"{row['mean_robust_offset']:.3f} | {100*row['coverage_delta6']:.1f}% | "
                f"{row['teacher_wer']:.2f} | {row['vanilla_wer']:.2f} | "
                f"{row['atdkd_delta6_wer']:.2f} | {row['rerr_delta6']:.2f}% | "
                f"{row['atdkd_delta9_wer']:.2f} | {row['rerr_delta9']:.2f}% | "
                f"{row['delta9_gain_over_delta6']:+.2f} |"
            )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")

    latex = []
    for scheme in schemes:
        latex += [
            f"% {scheme}",
            "\\begin{tabular}{llrrrrrrrrrrr}",
            "\\toprule",
            "Group & Mean-offset range & Utt. share & Mean offset & Cov.@$\\delta$=6 & Teacher & Vanilla & D6 & RERR & D9 & RERR & D9 gain \\\\",
            "\\midrule",
        ]
        for row in results:
            if row["scheme"] != scheme:
                continue
            frame_range = row["mean_offset_range"].replace(">=", "$\\geq$").replace("<", "$<$")
            latex.append(
                f"{row['group']} & {frame_range} & {100*row['utterance_share']:.1f}\\% & "
                f"{row['mean_robust_offset']:.3f} & {100*row['coverage_delta6']:.1f}\\% & "
                f"{row['teacher_wer']:.2f} & {row['vanilla_wer']:.2f} & "
                f"{row['atdkd_delta6_wer']:.2f} & {row['rerr_delta6']:.2f}\\% & "
                f"{row['atdkd_delta9_wer']:.2f} & {row['rerr_delta9']:.2f}\\% & "
                f"{row['delta9_gain_over_delta6']:+.2f} \\\\"
            )
        latex += ["\\bottomrule", "\\end{tabular}", ""]
    output.with_suffix(".tex").write_text("\n".join(latex))
    print(f"wrote {output}.csv/.md/.tex")


if __name__ == "__main__":
    main()
