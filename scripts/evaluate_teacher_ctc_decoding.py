#!/usr/bin/env python
"""Save greedy and no-LM beam predictions from a pretrained NeMo CTC model."""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from nemo.collections.asr.models import ASRModel

from evaluate_ctc_decoding import make_beam_decoder, normalize_text, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--beam-workers", type=int, default=8)
    parser.add_argument("--pyctcdecode-path", default="/tmp/span_kd_pyctcdecode")
    args = parser.parse_args()

    if args.pyctcdecode_path:
        sys.path.insert(0, args.pyctcdecode_path)
    model = ASRModel.restore_from(args.teacher, map_location="cpu").eval()
    if not hasattr(model, "blank_id"):
        model.blank_id = int(model.decoder.num_classes_with_blank - 1)
    decoder = make_beam_decoder(model, args.pyctcdecode_path)
    pool = mp.get_context("fork").Pool(args.beam_workers)
    model = model.to("cuda")

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    files = [row["audio_filepath"] for row in rows]
    references = [normalize_text(row["text"]) for row in rows]
    hypotheses = model.transcribe(
        files, batch_size=args.batch_size, return_hypotheses=True, verbose=False
    )
    greedy = [normalize_text(h.text) for h in hypotheses]
    emissions = [np.asarray(h.y_sequence.detach().float().cpu()) for h in hypotheses]
    beam = [normalize_text(text) for text in decoder.decode_batch(
        pool, emissions, beam_width=args.beam_width,
        beam_prune_logp=-10.0, token_min_logp=-5.0
    )]
    pool.close()
    pool.join()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row, ref, greedy_text, beam_text in zip(rows, references, greedy, beam):
            handle.write(json.dumps({
                "utterance_id": row.get("utterance_id", Path(row["audio_filepath"]).stem),
                "audio_filepath": row["audio_filepath"],
                "reference": ref,
                "greedy": greedy_text,
                "beam": beam_text,
            }) + "\n")
    print("greedy", summarize(greedy, references))
    print("beam", summarize(beam, references))
    print(f"predictions={args.output}")


if __name__ == "__main__":
    main()
