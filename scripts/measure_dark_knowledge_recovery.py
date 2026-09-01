#!/usr/bin/env python3
"""Measure held-out occurrence-level teacher dark-knowledge recovery.

The teacher cache is built from raw teacher posteriors pooled by transcript-
constrained delta-depeaked CTC occupancy.  Every student is then evaluated on
the same fixed teacher occurrence coordinate.  Reported metrics mirror the
hierarchical target used during training:

  * non-target mass MAE: coarse [blank, GT, all non-target] calibration;
  * teacher dark top-k recall: recovery of alternative-token identity;
  * top-M+tail JS: recovery of the conditional non-target distribution.

Teacher cache construction and student measurement are restartable.  Cache
files are keyed by split and manifest order and validate audio path, text, and
token ids before use.
"""

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ctc_fb import batched_ctc_token_gamma  # noqa: E402
from data import make_dataloader, read_manifest  # noqa: E402
from evaluate_student import load_model  # noqa: E402
from nemo.collections.asr.models import EncDecCTCModelBPE  # noqa: E402


def parse_named(values, kind):
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must use NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid {kind}: {value!r}")
        result.append((name, Path(path)))
    return result


def load_teacher(source):
    if source.endswith(".nemo"):
        return EncDecCTCModelBPE.restore_from(source)
    return EncDecCTCModelBPE.from_pretrained(source)


def loader(manifest, tokenizer, batch_size, workers):
    return make_dataloader(
        str(manifest), tokenizer, batch_size=batch_size, shuffle=False,
        sample_rate=16000, num_workers=workers, pin_memory=True,
    )


def cache_path(cache_dir, split, index):
    return cache_dir / split / f"{index:06d}.pt"


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def valid_cache(path, row, tokens, delta, top_m):
    if not path.exists():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return (
            value.get("audio_filepath") == row["audio_filepath"]
            and value.get("text") == row["text"]
            and torch.equal(value["tokens"].long(), tokens.cpu().long())
            and float(value.get("teacher_blank_penalty", -1.0)) == float(delta)
            and int(value.get("dark_top_m", -1)) == int(top_m)
        )
    except Exception:
        return False


@torch.no_grad()
def build_teacher_cache(teacher, manifests, cache_dir, delta, top_m,
                        batch_size, workers, device):
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    teacher = teacher.to(device).eval()
    summary = {}
    for split, manifest in manifests:
        rows = read_manifest(manifest)
        offset = 0
        built = 0
        reused = 0
        dl = loader(manifest, teacher.tokenizer, batch_size, workers)
        for batch_index, batch in enumerate(dl):
            batch_count = int(batch["wavs"].shape[0])
            paths = [cache_path(cache_dir, split, offset + b) for b in range(batch_count)]
            all_valid = all(
                valid_cache(
                    paths[b], rows[offset + b],
                    batch["tokens"][b, : int(batch["token_lens"][b])],
                    delta, top_m,
                )
                for b in range(batch_count)
            )
            if all_valid:
                reused += batch_count
                offset += batch_count
                if (batch_index + 1) % 25 == 0:
                    print(f"[teacher-cache:{split}] {offset}/{len(rows)} "
                          f"built={built} reused={reused}", flush=True)
                continue

            gpu = {k: v.to(device) if torch.is_tensor(v) else v
                   for k, v in batch.items()}
            log_probs, enc_len, _ = teacher.forward(
                input_signal=gpu["wavs"], input_signal_length=gpu["wav_lens"])
            gamma = batched_ctc_token_gamma(
                log_probs, enc_len, gpu["tokens"], gpu["token_lens"],
                blank, blank_penalty=delta)
            probs = log_probs.float().exp()

            for b in range(batch_count):
                n = int(gpu["token_lens"][b].item())
                frames = int(enc_len[b].item())
                tokens = gpu["tokens"][b, :n].long()
                support = gamma[b, :n, :frames].float()
                weights = support / support.sum(dim=1, keepdim=True).clamp_min(1e-12)
                pooled = torch.matmul(weights, probs[b, :frames].float())
                row_index = torch.arange(n, device=device)
                gt_mass = pooled[row_index, tokens]
                blank_mass = pooled[:, blank]
                nt_mass = (1.0 - blank_mass - gt_mass).clamp_min(0.0)
                mass3 = torch.stack([blank_mass, gt_mass, nt_mass], dim=1)
                mass3 = mass3 / mass3.sum(dim=1, keepdim=True).clamp_min(1e-12)

                conditional = pooled.clone()
                conditional[:, blank] = 0.0
                conditional[row_index, tokens] = 0.0
                conditional = conditional / conditional.sum(
                    dim=1, keepdim=True).clamp_min(1e-12)
                m = min(top_m, conditional.shape[1] - 2)
                dark_probs, dark_ids = conditional.topk(m, dim=1)
                dark_tail = (1.0 - dark_probs.sum(dim=1)).clamp_min(0.0)

                row = rows[offset + b]
                payload = {
                    "audio_filepath": row["audio_filepath"],
                    "text": row["text"],
                    "tokens": tokens.cpu().to(torch.int32),
                    "support": support.cpu().to(torch.float16),
                    "teacher_frames": frames,
                    "blank_id": blank,
                    "mass3_probs": mass3.cpu().to(torch.float16),
                    "dark_ids": dark_ids.cpu().to(torch.int32),
                    "dark_probs": dark_probs.cpu().to(torch.float16),
                    "dark_tail_prob": dark_tail.cpu().to(torch.float16),
                    "dark_top_m": m,
                    "teacher_blank_penalty": float(delta),
                    "mode": "heldout_teacher_dark_recovery",
                }
                atomic_torch_save(payload, paths[b])
                built += 1
            offset += batch_count
            if (batch_index + 1) % 25 == 0:
                print(f"[teacher-cache:{split}] {offset}/{len(rows)} "
                      f"built={built} reused={reused}", flush=True)
        if offset != len(rows):
            raise RuntimeError(f"cache count mismatch for {split}: {offset} != {len(rows)}")
        summary[split] = {"utterances": len(rows), "built": built, "reused": reused}
        print(f"[teacher-cache:{split}] done {summary[split]}", flush=True)
    return blank, summary


def student_grid_support(support, student_frames):
    """Match TransitionKDModel._span_ntdk_loss teacher->student resampling."""
    teacher_frames = int(support.shape[1])
    base = (torch.arange(student_frames, device=support.device, dtype=torch.float32)
            / max(student_frames, 1) * teacher_frames)
    shift = teacher_frames / max(student_frames, 1)
    idx_a = torch.floor(base + 0.25 * shift).long().clamp(0, teacher_frames - 1)
    idx_b = torch.floor(base + 0.75 * shift).long().clamp(0, teacher_frames - 1)
    result = 0.5 * (support[:, idx_a] + support[:, idx_b])
    return result / result.sum(dim=1, keepdim=True).clamp_min(1e-12)


def js_per_row(teacher, student):
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-12)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1e-12)
    midpoint = 0.5 * (teacher + student)
    t_term = torch.where(
        teacher > 0,
        teacher * (teacher.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()),
        torch.zeros_like(teacher),
    )
    s_term = torch.where(
        student > 0,
        student * (student.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()),
        torch.zeros_like(student),
    )
    return 0.5 * (t_term.sum(dim=1) + s_term.sum(dim=1))


def empty_stats():
    return defaultdict(float, {
        "tokens": 0.0,
        "nt_mass_abs_sum": 0.0,
        "top1_match_sum": 0.0,
        "top4_recall_sum": 0.0,
        "dark_js_sum": 0.0,
    })


def add_stats(target, source):
    for key, value in source.items():
        target[key] += float(value)


def finalize_stats(stats):
    count = max(float(stats["tokens"]), 1.0)
    return {
        "tokens": int(stats["tokens"]),
        "nt_mass_mae": stats["nt_mass_abs_sum"] / count,
        "dark_top1_agreement": stats["top1_match_sum"] / count,
        "dark_top4_recall": stats["top4_recall_sum"] / count,
        "dark_js_nats": stats["dark_js_sum"] / count,
        "dark_js_normalized": stats["dark_js_sum"] / count / math.log(2.0),
    }


@torch.no_grad()
def measure_student(label, checkpoint, config, manifests, cache_dir,
                    batch_size, workers, device):
    print(f"[student:{label}] loading {checkpoint}", flush=True)
    model, _ = load_model(str(config), str(checkpoint), device)
    model.eval()
    blank = int(model.blank_id)
    split_results = {}
    pooled_stats = empty_stats()

    for split, manifest in manifests:
        rows = read_manifest(manifest)
        stats = empty_stats()
        offset = 0
        dl = loader(manifest, model.tokenizer, batch_size, workers)
        for batch_index, batch in enumerate(dl):
            gpu = {k: v.to(device) if torch.is_tensor(v) else v
                   for k, v in batch.items()}
            log_probs, enc_len, _ = model.forward(
                input_signal=gpu["wavs"], input_signal_length=gpu["wav_lens"])
            student_probs = log_probs.float().exp()
            batch_count = int(gpu["wavs"].shape[0])

            for b in range(batch_count):
                cached = torch.load(
                    cache_path(cache_dir, split, offset + b),
                    map_location="cpu", weights_only=False)
                n = int(gpu["token_lens"][b].item())
                tokens = gpu["tokens"][b, :n].long()
                if cached["audio_filepath"] != rows[offset + b]["audio_filepath"]:
                    raise RuntimeError(f"cache/audio mismatch at {split}:{offset+b}")
                if not torch.equal(cached["tokens"].long(), tokens.cpu()):
                    raise RuntimeError(f"cache/token mismatch at {split}:{offset+b}")
                if int(cached["blank_id"]) != blank:
                    raise RuntimeError(
                        f"blank mismatch at {split}:{offset+b}: "
                        f"teacher={cached['blank_id']} student={blank}")

                frames = int(enc_len[b].item())
                support = cached["support"].to(device=device, dtype=torch.float32)
                support = student_grid_support(support, frames)
                pooled = torch.matmul(support, student_probs[b, :frames])
                row_index = torch.arange(n, device=device)

                teacher_mass3 = cached["mass3_probs"].to(
                    device=device, dtype=torch.float32)
                student_gt = pooled[row_index, tokens]
                student_blank = pooled[:, blank]
                student_nt = (1.0 - student_blank - student_gt).clamp_min(1e-12)
                mass_error = (student_nt - teacher_mass3[:, 2]).abs()

                dark_ids = cached["dark_ids"].to(device=device, dtype=torch.long)
                teacher_selected = cached["dark_probs"].to(
                    device=device, dtype=torch.float32)
                teacher_tail = cached["dark_tail_prob"].to(
                    device=device, dtype=torch.float32)
                student_selected = torch.gather(pooled, 1, dark_ids)
                student_tail = (student_nt - student_selected.sum(dim=1)).clamp_min(0.0)
                teacher_bins = torch.cat(
                    [teacher_selected, teacher_tail.unsqueeze(1)], dim=1)
                student_bins = torch.cat(
                    [student_selected, student_tail.unsqueeze(1)], dim=1)
                js = js_per_row(teacher_bins, student_bins)

                alternatives = pooled.clone()
                alternatives[:, blank] = -1.0
                alternatives[row_index, tokens] = -1.0
                student_top4 = alternatives.topk(4, dim=1).indices
                teacher_top4 = dark_ids[:, :4]
                top4_hits = (
                    teacher_top4.unsqueeze(2) == student_top4.unsqueeze(1)
                ).any(dim=2).float().sum(dim=1) / 4.0
                top1_match = (dark_ids[:, 0] == student_top4[:, 0]).float()

                stats["tokens"] += n
                stats["nt_mass_abs_sum"] += float(mass_error.sum().cpu())
                stats["top1_match_sum"] += float(top1_match.sum().cpu())
                stats["top4_recall_sum"] += float(top4_hits.sum().cpu())
                stats["dark_js_sum"] += float(js.sum().cpu())
            offset += batch_count
            if (batch_index + 1) % 25 == 0:
                print(f"[student:{label}:{split}] {offset}/{len(rows)}", flush=True)
        if offset != len(rows):
            raise RuntimeError(f"student count mismatch for {split}: {offset} != {len(rows)}")
        split_results[split] = finalize_stats(stats)
        add_stats(pooled_stats, stats)
        print(f"[student:{label}:{split}] {split_results[split]}", flush=True)

    split_results["pooled"] = finalize_stats(pooled_stats)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return split_results


def write_results(results, output_json, output_md):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    baseline = results["models"].get("mass3_only", {}).get("pooled", {})
    baseline_js = baseline.get("dark_js_nats")
    lines = [
        "# Held-out occurrence-level dark-knowledge recovery (seed 1)",
        "",
        "LibriSpeech test-clean and test-other token occurrences are micro-pooled. "
        "All students are measured on the same teacher delta=6 occurrence support.",
        "",
        "| Student | Tokens | NT-mass MAE (pp) ↓ | Dark Top-1 (%) ↑ | "
        "Dark Top-4 recall (%) ↑ | Dark JS (nats) ↓ | Gap closed vs Mass3 (%) ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, model_result in results["models"].items():
        value = model_result["pooled"]
        recovery = "—"
        if baseline_js and baseline_js > 0:
            recovery = f"{100.0 * (1.0 - value['dark_js_nats'] / baseline_js):.2f}"
        lines.append(
            f"| {label} | {value['tokens']:,} | "
            f"{100.0 * value['nt_mass_mae']:.3f} | "
            f"{100.0 * value['dark_top1_agreement']:.2f} | "
            f"{100.0 * value['dark_top4_recall']:.2f} | "
            f"{value['dark_js_nats']:.6f} | {recovery} |"
        )
    lines.extend([
        "",
        "`Gap closed vs Mass3 = 100 × (1 − JS_model / JS_mass3-only)`. "
        "Negative values mean a larger conditional dark-distribution gap than Mass3-only.",
        "",
    ])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True,
                        help="NAME=PATH; repeat for test-clean/test-other")
    parser.add_argument("--model", action="append", required=True,
                        help="LABEL=CHECKPOINT; repeat for students")
    parser.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    parser.add_argument("--config", default="configs/student_base.yaml")
    parser.add_argument("--cache-dir", default="analysis/dark_recovery_teacher_cache_d6_m32")
    parser.add_argument("--output-json", default="analysis/dark_knowledge_recovery_seed1.json")
    parser.add_argument("--output-md", default="analysis/dark_knowledge_recovery_seed1.md")
    parser.add_argument("--delta", type=float, default=6.0)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--student-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifests = parse_named(args.manifest, "manifest")
    models = parse_named(args.model, "model")
    for _, path in manifests + models:
        if not path.exists():
            raise FileNotFoundError(path)

    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device)
    print(f"[teacher] loading {args.teacher}", flush=True)
    teacher = load_teacher(args.teacher)
    blank, cache_summary = build_teacher_cache(
        teacher, manifests, cache_dir, args.delta, args.top_m,
        args.teacher_batch_size, args.workers, device)
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "protocol": {
            "teacher": args.teacher,
            "teacher_blank_penalty": args.delta,
            "dark_top_m": args.top_m,
            "teacher_blank_id": blank,
            "manifests": {name: str(path) for name, path in manifests},
            "cache_summary": cache_summary,
            "averaging": "micro-average over token occurrences",
            "support": "fixed teacher d6 transcript-constrained CTC occupancy",
            "student_pooling": "same two-point teacher-to-student grid resampling as training",
        },
        "checkpoints": {name: str(path) for name, path in models},
        "models": {},
    }
    for label, checkpoint in models:
        results["models"][label] = measure_student(
            label, checkpoint, Path(args.config), manifests, cache_dir,
            args.student_batch_size, args.workers, device)
        write_results(results, Path(args.output_json), Path(args.output_md))

    write_results(results, Path(args.output_json), Path(args.output_md))
    print(f"[done] {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
