#!/usr/bin/env python3
"""Summarize the matched seed-1 delta=0/6/9 no-LM beam evaluation."""

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_beam(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    current = None
    scores: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.fullmatch(r"\[([^]]+)\]", line.strip())
        if match:
            current = match.group(1)
            continue
        match = re.match(r"beam\s+WER=([0-9.]+)%", line.strip())
        if match and current:
            scores[current] = float(match.group(1))
    return scores


def pick(*candidates: str) -> Path:
    for candidate in candidates:
        path = ROOT / candidate
        if path.is_file():
            return path
    raise FileNotFoundError("none of these evaluation logs exists: " + ", ".join(candidates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=16)
    args = parser.parse_args()
    beam = args.beam_width

    logs = {
        0: (
            pick(f"analysis/beam{beam}_ablation_lbs_full_d0.txt"),
            pick(f"analysis/beam{beam}_ablation_chime3_full_d0.txt"),
        ),
        6: (
            pick(
                f"analysis/beam{beam}_main3seed_lbs_ours_s1.txt",
                f"analysis/beam{beam}_paper_lbs_mass25_ntdk8.txt",
            ),
            pick(
                f"analysis/beam{beam}_main3seed_chime3_ours_s1.txt",
                f"analysis/beam{beam}_paper_chime3_lr05_mass25_ntdk8.txt",
            ),
        ),
        9: (
            pick(f"analysis/beam{beam}_ablation_lbs_full_d9_s1.txt"),
            pick(f"analysis/beam{beam}_ablation_chime3_full_d9_s1.txt"),
        ),
    }
    lbs_splits = ("dev_clean", "dev_other", "test_clean", "test_other")
    ch_splits = ("chime3_dev_real", "chime3_dev_simu", "chime3_eval_real", "chime3_eval_simu")

    rows = []
    for delta, (lbs_log, ch_log) in logs.items():
        lbs = parse_beam(lbs_log)
        chime = parse_beam(ch_log)
        missing = [s for s in (*lbs_splits, *ch_splits) if s not in (lbs if s in lbs_splits else chime)]
        if missing:
            raise RuntimeError(f"missing beam results for delta={delta}: {missing}")
        rows.append((delta, *(lbs[s] for s in lbs_splits), *(chime[s] for s in ch_splits)))

    out = ROOT / "analysis" / f"delta_0_6_9_ablation_beam{beam}.md"
    header = (
        f"# Delta ablation (seed 1, no-LM beam-{beam})\n\n"
        "WER (%), lower is better. All settings use Mass3 weight 25, NTDK weight 8, and M=32.\n\n"
        "| Delta | LS Dev Clean | LS Dev Other | LS Test Clean | LS Test Other | "
        "CH Dev Real | CH Dev Sim. | CH Eval Real | CH Eval Sim. |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        "| " + str(row[0]) + " | " + " | ".join(f"{value:.2f}" for value in row[1:]) + " |\n"
        for row in rows
    )
    out.write_text(header + body)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
