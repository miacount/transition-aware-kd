"""Merge frame-KD and token-avg manifests into a single combined manifest.
Joins on audio_filepath; only rows present in both are kept.
"""
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame-kd",   required=True)
    p.add_argument("--token-avg",  required=True)
    p.add_argument("--out",        required=True)
    args = p.parse_args()

    def load(path):
        rows = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows[r["audio_filepath"]] = r
        return rows

    frame_rows = load(args.frame_kd)
    tavg_rows  = load(args.token_avg)

    common = set(frame_rows) & set(tavg_rows)
    print(f"frame_kd rows : {len(frame_rows)}")
    print(f"token_avg rows: {len(tavg_rows)}")
    print(f"combined rows : {len(common)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for key in sorted(common):
            row = dict(frame_rows[key])
            row["teacher_token_avg_path"] = tavg_rows[key]["teacher_token_avg_path"]
            if "teacher_target" in tavg_rows[key]:
                row["teacher_target"] = tavg_rows[key]["teacher_target"]
            f.write(json.dumps(row) + "\n")

    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
