#!/usr/bin/env python
"""Greedy error breakdown and no-LM prefix-beam evaluation for CTC models."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_student import load_model, make_split_cfgs  # noqa: E402

from nemo.collections.asr.metrics.wer import word_error_rate_detail  # noqa: E402


def normalize_text(text):
    return " ".join(text.split())


def collapse_greedy(token_ids, blank_id):
    output = []
    previous = -1
    for token in token_ids:
        token = int(token)
        if token != blank_id and token != previous:
            output.append(token)
        previous = token
    return output


def make_beam_decoder(model, dependency_path):
    if dependency_path:
        sys.path.insert(0, dependency_path)
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise RuntimeError(
            "pyctcdecode is required for --beam-width > 0. Install it or pass "
            "--pyctcdecode-path pointing to an installation."
        ) from exc

    labels = list(model.tokenizer.vocab)
    if model.tokenizer.unk_id is not None and model.tokenizer.unk_id >= 0:
        # pyctcdecode's BPE decoder recognizes this spelling as SentencePiece UNK.
        labels[model.tokenizer.unk_id] = "▁⁇▁"
    if model.blank_id == len(labels):
        labels.append("")
    else:
        labels.insert(model.blank_id, "")
    return build_ctcdecoder(labels=labels, kenlm_model_path=None)


def summarize(hypotheses, references):
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=hypotheses, references=references, use_cer=False
    )
    return {
        "wer": wer,
        "words": words,
        "ins_rate": ins_rate,
        "del_rate": del_rate,
        "sub_rate": sub_rate,
        "insertions": int(round(ins_rate * words)),
        "deletions": int(round(del_rate * words)),
        "substitutions": int(round(sub_rate * words)),
    }


def print_summary(label, result):
    print(
        f"{label:8s} WER={100.0 * result['wer']:.4f}% "
        f"INS={100.0 * result['ins_rate']:.4f}%({result['insertions']}) "
        f"DEL={100.0 * result['del_rate']:.4f}%({result['deletions']}) "
        f"SUB={100.0 * result['sub_rate']:.4f}%({result['substitutions']}) "
        f"REF_WORDS={result['words']}"
    )


@torch.no_grad()
def evaluate_split(model, ds_cfg, device, decoder, pool, args):
    dataloader = model._make_dataloader_from_cfg(ds_cfg, shuffle=False)
    references = []
    greedy_hypotheses = []
    beam_hypotheses = []
    total_loss = 0.0
    total_items = 0
    started = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        log_probs, enc_len, _ = model.forward(
            input_signal=batch["wavs"], input_signal_length=batch["wav_lens"]
        )
        loss = model.loss(
            log_probs=log_probs,
            targets=batch["tokens"],
            input_lengths=enc_len,
            target_lengths=batch["token_lens"],
        )

        remaining = None
        if args.max_utts is not None:
            remaining = args.max_utts - len(references)
            if remaining <= 0:
                break
        take = log_probs.shape[0] if remaining is None else min(log_probs.shape[0], remaining)
        total_loss += float(loss.detach().cpu()) * take
        total_items += take

        lengths = enc_len[:take].detach().cpu().tolist()
        argmax_ids = log_probs[:take].argmax(dim=-1).detach().cpu()
        emissions = []
        for index, length in enumerate(lengths):
            target_length = int(batch["token_lens"][index].item())
            reference_ids = batch["tokens"][index, :target_length].detach().cpu().tolist()
            reference = normalize_text(model.tokenizer.ids_to_text(reference_ids))
            greedy_ids = collapse_greedy(argmax_ids[index, :length].tolist(), model.blank_id)
            greedy = normalize_text(model.tokenizer.ids_to_text(greedy_ids))
            references.append(reference)
            greedy_hypotheses.append(greedy)
            if decoder is not None:
                emissions.append(
                    log_probs[index, :length].detach().float().cpu().numpy()
                )

        if decoder is not None:
            decoded = decoder.decode_batch(
                pool,
                emissions,
                beam_width=args.beam_width,
                beam_prune_logp=args.beam_prune_logp,
                token_min_logp=args.token_min_logp,
            )
            beam_hypotheses.extend(normalize_text(text) for text in decoded)

        if (batch_idx + 1) % args.progress_every == 0:
            print(
                f"  progress utterances={len(references)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        if args.max_utts is not None and len(references) >= args.max_utts:
            break

    result = {
        "utterances": len(references),
        "ctc_loss": total_loss / max(total_items, 1),
        "greedy": summarize(greedy_hypotheses, references),
    }
    if decoder is not None:
        result["beam"] = summarize(beam_hypotheses, references)
    result["predictions"] = {
        "references": references,
        "greedy": greedy_hypotheses,
        "beam": beam_hypotheses if decoder is not None else [],
    }
    return result


def write_predictions(output_dir, split_name, manifest_path, model_name, predictions):
    manifest_rows = [json.loads(line) for line in Path(manifest_path).read_text().splitlines()
                     if line.strip()]
    references = predictions["references"]
    if len(manifest_rows) < len(references):
        raise RuntimeError(
            f"manifest has {len(manifest_rows)} rows but decoding produced {len(references)} predictions"
        )
    output_path = Path(output_dir) / f"{model_name}_{split_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for index, reference in enumerate(references):
            source = manifest_rows[index]
            row = {
                "utterance_id": source.get("utterance_id", Path(source["audio_filepath"]).stem),
                "audio_filepath": source["audio_filepath"],
                "reference": reference,
                "greedy": predictions["greedy"][index],
            }
            if predictions["beam"]:
                row["beam"] = predictions["beam"][index]
            handle.write(json.dumps(row) + "\n")
    print(f"predictions={output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--name", default="model")
    parser.add_argument("--config", default="configs/student_base.yaml")
    parser.add_argument("--manifest", action="append", required=True,
                        help="name=path; repeat for multiple dev sets")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam-width", type=int, default=16,
                        help="0 disables beam decoding")
    parser.add_argument("--beam-prune-logp", type=float, default=-10.0)
    parser.add_argument("--token-min-logp", type=float, default=-5.0)
    parser.add_argument("--beam-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--pyctcdecode-path", default="/tmp/span_kd_pyctcdecode")
    parser.add_argument("--max-utts", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--predictions-dir", default="",
                        help="optional directory for per-utterance JSONL hypotheses")
    args = parser.parse_args()

    pool = None
    if args.beam_width > 0 and args.pyctcdecode_path:
        # Workers must inherit this path before the decoder's bound method is
        # unpickled in them.
        sys.path.insert(0, args.pyctcdecode_path)
    try:
        device = torch.device(args.device)
        # Construct the decoder and its global model container before forking.
        # Load on CPU first so CUDA is still uninitialized at fork time.
        model, cfg = load_model(args.config, args.ckpt, torch.device("cpu"))
        decoder = make_beam_decoder(model, args.pyctcdecode_path) if args.beam_width > 0 else None
        if decoder is not None and args.beam_workers > 1:
            # decode_batch rejects spawn pools and relies on decoder state
            # inherited by forked workers.
            pool = mp.get_context("fork").Pool(args.beam_workers)
        model = model.to(device)
        print(f"model={args.name}")
        print(f"checkpoint={args.ckpt}")
        if decoder is not None:
            print(
                f"beam_width={args.beam_width} beam_prune_logp={args.beam_prune_logp} "
                f"token_min_logp={args.token_min_logp} workers={args.beam_workers}"
            )
        for ds_cfg in make_split_cfgs(cfg, args.manifest):
            split_name = ds_cfg.get(
                "name", os.path.splitext(os.path.basename(ds_cfg.manifest_filepath))[0]
            )
            print(f"\n[{split_name}]", flush=True)
            result = evaluate_split(model, ds_cfg, device, decoder, pool, args)
            print(f"utterances={result['utterances']} CTC_loss={result['ctc_loss']:.4f}")
            print_summary("greedy", result["greedy"])
            if "beam" in result:
                print_summary("beam", result["beam"])
                print(
                    f"beam_minus_greedy_WER={100.0 * (result['beam']['wer'] - result['greedy']['wer']):+.4f}%p"
                )
            if args.predictions_dir:
                write_predictions(
                    args.predictions_dir,
                    split_name,
                    ds_cfg.manifest_filepath,
                    args.name,
                    result["predictions"],
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
