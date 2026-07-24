#!/usr/bin/env python
"""Greedy corpus WER of a pretrained NeMo teacher on manifest(s).

Used to establish the KD ceiling on a new dataset before building targets
(cf. METHOD_REPORT §0: teacher 3.70/8.14 on LibriSpeech test-clean/other).

Usage:
  python scripts/evaluate_teacher.py \
      --manifest dev=data/tedlium_dev.json --manifest test=data/tedlium_test.json
"""
import argparse
import json

import torch
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--manifest", action="append", required=True,
                    help="name=path, repeatable")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASRModel.from_pretrained(args.teacher, map_location=device)
    model.eval()

    for spec in args.manifest:
        name, path = spec.split("=", 1)
        files, refs = [], []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                files.append(row["audio_filepath"])
                refs.append(row["text"])
        with torch.inference_mode():
            hyps = model.transcribe(files, batch_size=args.batch_size, verbose=False)
        texts = [h.text if hasattr(h, "text") else h for h in hyps]
        wer = word_error_rate(hypotheses=texts, references=refs)
        print(f"{name:12s} utts={len(refs):6d}  WER={wer * 100:.2f}%")


if __name__ == "__main__":
    main()
