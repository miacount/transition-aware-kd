#!/usr/bin/env python
"""Download LibriSpeech eval splits and build NeMo JSON manifests."""
import argparse
import tarfile
import urllib.request
from pathlib import Path

from build_librispeech_manifest import build_manifest


URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}


def manifest_name(split):
    return split.replace("-", "_") + ".json"


def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"exists    : {out_path}")
        return
    print(f"download  : {url}")
    print(f"to        : {out_path}")
    urllib.request.urlretrieve(url, out_path)


def extract(archive, data_dir):
    split = archive.name.replace(".tar.gz", "")
    split_dir = data_dir / "LibriSpeech" / split
    if split_dir.exists():
        print(f"extracted : {split_dir}")
        return split_dir
    print(f"extract   : {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    return split_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["dev-clean", "dev-other", "test-clean", "test-other"],
        choices=sorted(URLS),
    )
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--keep_case", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    for split in args.splits:
        archive = data_dir / f"{split}.tar.gz"
        download(URLS[split], archive)
        split_dir = extract(archive, data_dir)
        manifest_out = data_dir / manifest_name(split)
        rows = build_manifest(
            split_dir=split_dir,
            manifest_out=manifest_out,
            processed_dir=None,
            lowercase=not args.keep_case,
        )
        hours = sum(r["duration"] for r in rows) / 3600.0
        print(f"manifest  : {manifest_out} ({len(rows)} utts, {hours:.2f} h)")


if __name__ == "__main__":
    main()
