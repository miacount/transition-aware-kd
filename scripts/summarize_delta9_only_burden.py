#!/usr/bin/env python
"""Test whether delta=9 helps where it uniquely recovers teacher support."""

import csv
import json
from pathlib import Path

import numpy as np

from summarize_natural_mismatch import edit_distance


EPS = 0.01
BOOTSTRAP = 5000
GROUPS = (
    ("Low (<=10%)", lambda x: x <= 0.10),
    ("Mid (10-20%)", lambda x: 0.10 < x <= 0.20),
    ("High (>20%)", lambda x: x > 0.20),
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prediction_counts(path):
    return {
        row["utterance_id"]: edit_distance(row["reference"], row["beam"])
        for row in read_jsonl(path)
    }


def bootstrap_delta(ids, d6, d9, rng):
    d6_edits = np.array([d6[utt][0] for utt in ids])
    d9_edits = np.array([d9[utt][0] for utt in ids])
    words = np.array([d6[utt][1] for utt in ids])
    values = np.empty(BOOTSTRAP)
    for rep in range(BOOTSTRAP):
        index = rng.integers(0, len(ids), len(ids))
        values[rep] = 100.0 * (d6_edits[index].sum() - d9_edits[index].sum()) / words[index].sum()
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def main():
    rng = np.random.default_rng(20260828)
    results = []
    for split in ("test_clean", "test_other"):
        coverage_root = Path(f"analysis/natural_mismatch_{split}_delta9")
        prediction_root = Path(f"analysis/natural_mismatch_{split}/predictions")
        rows = read_jsonl(coverage_root / "per_utterance_no_kd.jsonl")
        arrays = np.load(coverage_root / "offsets.npz")
        hit6 = arrays["no_kd_cov6"] > EPS
        hit9 = arrays["no_kd_cov9"] > EPS
        burden = {}
        cursor = 0
        for row in rows:
            stop = cursor + row["n_tokens"]
            burden[row["utterance_id"]] = float((hit9[cursor:stop] & ~hit6[cursor:stop]).mean())
            cursor = stop
        assert cursor == len(hit6)

        vanilla = prediction_counts(prediction_root / f"vanilla_s1_{split}.jsonl")
        d6_name = f"atdkd_s1_{split}.jsonl" if split == "test_clean" else f"atdkd_d6_s1_{split}.jsonl"
        d6 = prediction_counts(prediction_root / d6_name)
        d9 = prediction_counts(prediction_root / f"atdkd_d9_s1_{split}.jsonl")
        assert set(burden) == set(vanilla) == set(d6) == set(d9)

        for label, predicate in GROUPS:
            ids = [utt for utt, value in burden.items() if predicate(value)]
            words = sum(vanilla[utt][1] for utt in ids)
            v_wer = 100.0 * sum(vanilla[utt][0] for utt in ids) / words
            d6_wer = 100.0 * sum(d6[utt][0] for utt in ids) / words
            d9_wer = 100.0 * sum(d9[utt][0] for utt in ids) / words
            delta = d6_wer - d9_wer
            low, high = bootstrap_delta(ids, d6, d9, rng)
            results.append({
                "split": split,
                "burden_group": label,
                "utterances": len(ids),
                "utterance_share": len(ids) / len(rows),
                "mean_delta9_only_burden": float(np.mean([burden[utt] for utt in ids])),
                "vanilla_wer": v_wer,
                "delta6_wer": d6_wer,
                "delta6_rerr": 100.0 * (v_wer - d6_wer) / v_wer,
                "delta9_wer": d9_wer,
                "delta9_rerr": 100.0 * (v_wer - d9_wer) / v_wer,
                "delta9_gain_over_delta6": delta,
                "delta9_gain_ci_low": low,
                "delta9_gain_ci_high": high,
            })

    output = Path("analysis/delta9_only_burden")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# Delta=9-only recoverable burden",
        "",
        "Burden is the utterance-level fraction of reference tokens hit by delta=9 but not delta=6 "
        "at occupancy > 0.01. Groups (<=10%, 10-20%, >20%) were fixed from burden distributions "
        "before loading WER predictions. All results are matched seed 1, no-LM beam-16. Positive "
        "D9 gain means delta=9 is better; CIs use 5,000 paired utterance bootstrap replicates.",
        "",
        "| Split | D9-only burden | Utt. share | Mean burden | Vanilla | AT-DKD d=6 | RERR d=6 | AT-DKD d=9 | RERR d=9 | D9 gain [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['split'].replace('_', '-')} | {row['burden_group']} | "
            f"{100*row['utterance_share']:.1f}% ({row['utterances']}) | "
            f"{100*row['mean_delta9_only_burden']:.1f}% | {row['vanilla_wer']:.2f} | "
            f"{row['delta6_wer']:.2f} | {row['delta6_rerr']:.2f}% | "
            f"{row['delta9_wer']:.2f} | {row['delta9_rerr']:.2f}% | "
            f"{row['delta9_gain_over_delta6']:+.2f} "
            f"[{row['delta9_gain_ci_low']:+.2f}, {row['delta9_gain_ci_high']:+.2f}] |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")

    latex = [
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Split & D9-only burden & Share & Mean & Vanilla & D6 & RERR & D9 & RERR & D9 gain \\\\",
        "\\midrule",
    ]
    for row in results:
        latex.append(
            f"{row['split'].replace('_', '-')} & {row['burden_group']} & "
            f"{100*row['utterance_share']:.1f}\\% & {100*row['mean_delta9_only_burden']:.1f}\\% & "
            f"{row['vanilla_wer']:.2f} & {row['delta6_wer']:.2f} & {row['delta6_rerr']:.2f}\\% & "
            f"{row['delta9_wer']:.2f} & {row['delta9_rerr']:.2f}\\% & "
            f"{row['delta9_gain_over_delta6']:+.2f} \\\\"
        )
    latex += ["\\bottomrule", "\\end{tabular}"]
    output.with_suffix(".tex").write_text("\n".join(latex) + "\n")
    print(f"wrote {output}.csv/.md/.tex")


if __name__ == "__main__":
    main()
