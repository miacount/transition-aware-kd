#!/usr/bin/env python
"""Attach an expanded-delta support to a fixed semantic-core manifest."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-manifest", required=True)
    ap.add_argument("--expanded-manifest", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--delta", type=float, required=True)
    args = ap.parse_args()

    core = read_rows(args.core_manifest)
    expanded = read_rows(args.expanded_manifest)
    if len(core) != len(expanded):
        raise ValueError(f"row-count mismatch: {len(core)} vs {len(expanded)}")

    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, (c, e) in enumerate(zip(core, expanded)):
            if c.get("audio_filepath") != e.get("audio_filepath") or c.get("text") != e.get("text"):
                raise ValueError(f"manifest alignment mismatch at row {i}")
            if "teacher_span_kd_path" not in c or "teacher_span_kd_path" not in e:
                raise ValueError(f"missing span target at row {i}")
            row = dict(c)
            row["teacher_span_shoulder_path"] = e["teacher_span_kd_path"]
            row["span_shoulder_delta"] = args.delta
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(core)} rows to {out_path} (delta={args.delta:g})")


if __name__ == "__main__":
    main()
