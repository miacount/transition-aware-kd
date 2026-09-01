#!/usr/bin/env python
"""Collect paper-suite decoding logs into reproducible Markdown and CSV tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METHODS = [
    ("No KD", "nokd"),
    ("Vanilla frame KD <sup>[1]</sup>", "vanilla_full_l025", "vanilla_full_l09"),
    ("Blank elimination KD <sup>[2]</sup>", "kdbe_full_l025", "kdbe_full_l09"),
    ("Symmetric selection KD <sup>[3]</sup>", "symmetric_full_l025_n4", "symmetric_full_l1_n2"),
    ("Guided CTC <sup>[4]</sup>", "guided_exact"),
    ("S-CTC + CTC fine-tune <sup>[5]</sup>", "sctc_ft"),
    ("CR-CTC <sup>[6]</sup>", "crctc_stable_fair50", "crctc_fair"),
    ("Mass3 + NTDK (ours) <sup>[7]</sup>", "mass25_ntdk8"),
]

SPLIT_RE = re.compile(r"^\[([^]]+)\]\s*$")
WER_RE = re.compile(r"^(greedy|beam)\s+WER=([0-9]+(?:\.[0-9]+)?)%")


def method_suffix(method: tuple[str, ...], dataset: str) -> str:
    if len(method) == 2:
        return method[1]
    return method[1] if dataset == "lbs" else method[2]


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    split = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        split_match = SPLIT_RE.match(line)
        if split_match:
            split = split_match.group(1)
            metrics.setdefault(split, {})
            continue
        wer_match = WER_RE.match(line)
        if split is not None and wer_match:
            metrics[split][wer_match.group(1)] = float(wer_match.group(2))
    return metrics


def value(results, method, dataset, split, decoder):
    suffix = method_suffix(method, dataset)
    return results.get((dataset, suffix), {}).get(split, {}).get(decoder)


def fmt(number: float | None, best: bool = False) -> str:
    if number is None:
        return "—"
    rendered = f"{number:.2f}"
    return f"**{rendered}**" if best else rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    missing = []
    for method in METHODS:
        for dataset in ("lbs",):
            suffix = method_suffix(method, dataset)
            path = args.analysis_dir / f"beam{args.beam_width}_paper_{dataset}_{suffix}.txt"
            if not path.is_file():
                missing.append(path)
                continue
            parsed = parse_log(path)
            if not parsed:
                raise SystemExit(f"No WER records found in {path}")
            results[(dataset, suffix)] = parsed

    if missing and not args.allow_missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Missing {len(missing)} evaluation logs:\n{rendered}")

    columns = [
        ("Greedy dev-clean", "dev_clean", "greedy"),
        ("Greedy dev-other", "dev_other", "greedy"),
        ("Greedy test-clean", "test_clean", "greedy"),
        ("Greedy test-other", "test_other", "greedy"),
        (f"Beam-{args.beam_width} test-clean", "test_clean", "beam"),
        (f"Beam-{args.beam_width} test-other", "test_other", "beam"),
    ]
    best = {}
    for title, split, decoder in columns:
        observed = [value(results, method, "lbs", split, decoder) for method in METHODS]
        observed = [number for number in observed if number is not None]
        best[title] = min(observed) if observed else None

    md_path = args.analysis_dir / f"paper_main_table_beam{args.beam_width}.md"
    lines = [
        "# LibriSpeech final results",
        "",
        "WER (%); lower is better. Greedy is reported on all dev/test splits; no-LM CTC beam decoding is reported on test only.",
        "Checkpoints are selected by minimum dev-clean greedy WER. Bold denotes the best observed value in each column.",
        "",
        f"| Method | dev-clean<br>Greedy | dev-other<br>Greedy | test-clean<br>Greedy | test-other<br>Greedy | test-clean<br>Beam-{args.beam_width} | test-other<br>Beam-{args.beam_width} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for title, split, decoder in columns:
            number = value(results, method, "lbs", split, decoder)
            cells.append(fmt(number, number is not None and number == best[title]))
        lines.append(f"| {method[0]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Reproduction notes",
        "",
        "1. **Vanilla frame KD** — Hinton et al., *Distilling the Knowledge in a Neural Network* (2015); the CTC/TED-style loss recipe follows Hilmes et al. (Interspeech 2025).",
        "2. **Blank elimination KD** — Tian et al., *Knowledge Distillation For CTC-based Speech Recognition Via Consistent Acoustic Representation Learning (CARL)* (Interspeech 2022); dataset-specific lambda follows Hilmes et al. (2025).",
        "3. **Symmetric selection KD** — Hilmes et al., *Analyzing the Importance of Blank for CTC-Based Knowledge Distillation* (Interspeech 2025).",
        "4. **Guided CTC** — Kurata and Audhkhasi, *Guiding CTC Posterior Spike Timings for Improved Posterior Fusion and Knowledge Distillation* (Interspeech 2019). Exact selected-posterior objective `-sum p_S`, weight 1.",
        "5. **S-CTC + CTC fine-tune** — Huang et al., *Knowledge Distillation for Sequence Model* (Interspeech 2018). Transcript-constrained teacher FB occupancy followed by supervised CTC fine-tuning; local compute-matched schedule is 80+20 epochs.",
        "6. **CR-CTC** — Yao et al., *Consistency Regularization for CTC-based Speech Recognition* (ICLR 2025). Faithful two-view consistency loss; the table uses the stable local 50-epoch recipe (physical batch 32, time-mask factor 1.5), not the collapsed strict-mask run.",
        "7. **Mass3 + NTDK** — proposed occurrence-aware hierarchical CTC distillation; `L_CTC + 25 L_Mass3 + 8 L_NTDK`.",
    ]
    if missing:
        lines += ["", f"> Incomplete table: {len(missing)} evaluation log(s) were unavailable."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = args.analysis_dir / f"paper_main_metrics_beam{args.beam_width}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "dataset", "split", "decoder", "wer_percent", "log"])
        for method in METHODS:
            suffix = method_suffix(method, "lbs")
            log_path = args.analysis_dir / f"beam{args.beam_width}_paper_lbs_{suffix}.txt"
            for _, split, decoder in columns:
                number = value(results, method, "lbs", split, decoder)
                if number is not None:
                    writer.writerow([method[0], "lbs", split, decoder, f"{number:.4f}", log_path])

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
