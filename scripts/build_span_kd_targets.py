#!/usr/bin/env python3
"""Build temporal-support span KD targets.

For each utterance, store:
  support:   (N_bpe, T_teacher) phoneme-derived BPE support a(t,u)
  avg_ids:   (N_bpe, top_k) teacher semantic top-k ids q_T,u
  avg_probs: (N_bpe, top_k) teacher semantic top-k probabilities
  gates:     (N_bpe,) reliability coverage gate

By default, the teacher provides semantic distributions and the phoneme aligner
provides only the temporal support coordinate. With
--teacher-target-weighting gamma_support, the teacher semantic target is also
localized by the aligner support.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phoneme_utils import load_lexicon, text_to_phone_ids  # noqa: E402
from train_soft_aligner import _wav_to_mel  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402
from visualize_temporal_support_kd import (  # noqa: E402
    load_phoneme_aligner,
    map_bpe_to_phone_spans,
    normalize_cols,
    resample_time,
)


def read_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_audio(path, sample_rate):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav


@torch.no_grad()
def run_teacher(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


@torch.no_grad()
def run_phone_model(model, wav_np, log_priors, alpha, device):
    mel = _wav_to_mel(wav_np, 16000).to(device)
    log_probs, out_len = model(mel.unsqueeze(0), torch.tensor([mel.shape[0]], device=device))
    frames = int(out_len[0].item())
    raw_lp = log_probs[0, :frames].detach().cpu().float().numpy()
    return raw_lp - alpha * log_priors.reshape(1, -1)


def blank_penalty_logprobs(log_probs, blank_id, delta):
    """Subtract delta (nats) from blank log-prob only; renormalize per frame.
    Non-blank ratios are preserved exactly (dark knowledge kept)."""
    if delta == 0.0:
        return log_probs
    lp = log_probs.copy()
    lp[:, blank_id] = log_probs[:, blank_id] - delta
    m = np.max(lp, axis=-1, keepdims=True)
    lp = lp - (m + np.log(np.exp(lp - m).sum(axis=-1, keepdims=True)))
    return lp


def teacher_semantic_topk(teacher_lp, teacher_gamma, blank_id, top_k, support=None,
                          keep_blank=False):
    probs = np.exp(teacher_lp)
    out_ids = []
    out_probs = []
    for u in range(teacher_gamma.shape[1]):
        weights = teacher_gamma[:, u]
        if support is not None:
            weights = weights * support[:, u]
            weights = weights / max(float(weights.sum()), 1e-12)
        q = (weights[:, None] * probs).sum(axis=0)
        if not keep_blank:
            q[blank_id] = 0.0   # default: drop blank, keep only "which token"
        # keep_blank=True: leave blank in q so the target also encodes
        # "how present vs gap" (span-level occupancy) for this token.
        q = q / max(float(q.sum()), 1e-12)
        k = min(top_k, q.shape[0])
        idx = np.argpartition(q, -k)[-k:]
        idx = idx[np.argsort(q[idx])[::-1]]
        vals = q[idx]
        vals = vals / max(float(vals.sum()), 1e-12)
        out_ids.append(idx.astype(np.int32))
        out_probs.append(vals.astype(np.float32))
    return np.stack(out_ids), np.stack(out_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-in", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--phoneme-ckpt", default="nemo_experiments/phoneme_soft_aligner/epoch_010.pt")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--out-dir", default="data/span_kd")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--span-mode", choices=["proportional", "word"], default="proportional")
    ap.add_argument(
        "--teacher-target-weighting",
        choices=["gamma", "gamma_support"],
        default="gamma",
        help="weight teacher semantic target by gamma only, or by normalized gamma * aligner support",
    )
    ap.add_argument(
        "--unk-policy",
        choices=["skip", "drop"],
        default="drop",
        help="skip: discard the whole utterance if any word is OOV. "
             "drop: drop only the OOV word (zero temporal support for its BPE tokens), keep the rest.",
    )
    ap.add_argument(
        "--teacher-only",
        action="store_true",
        help="Build support directly from the teacher's own transcript-constrained CTC "
             "forward-backward gamma, de-peaked by --teacher-blank-penalty. No phoneme "
             "aligner, no BPE->phoneme mapping. WHERE=de-peaked gamma, WHAT=sharp teacher "
             "posterior weighted by that gamma.",
    )
    ap.add_argument(
        "--teacher-blank-penalty", type=float, default=0.0,
        help="teacher-only mode: nats subtracted from blank log-prob before FB (0=peaky).",
    )
    ap.add_argument(
        "--keep-blank", action="store_true",
        help="keep blank in the per-token semantic target top-k (target encodes "
             "'which token + how present vs gap'), instead of dropping it.",
    )
    ap.add_argument(
        "--where-model", default="",
        help="teacher-only mode: path to a fine-tuned .nemo used ONLY for the support "
             "(WHERE) gamma. The original --teacher still provides the sharp semantic "
             "top-k (WHAT). If empty, the original teacher is used for both.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="reuse existing per-utterance .pt targets")
    ap.add_argument("--flush-every", type=int, default=100, help="write partial manifest every N kept rows")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[:args.limit]

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    bpe_blank = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    where_model = None
    if args.teacher_only:
        phone_model = phone_to_id = phone_blank = phone_alpha = phone_log_priors = lexicon = None
        print(f"[teacher-only] blank_penalty={args.teacher_blank_penalty} nats, no phoneme aligner")
        if args.where_model:
            where_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.where_model).to(args.device).eval()
            where_model.freeze()
            print(f"[teacher-only] WHERE source = fine-tuned {args.where_model}; WHAT = original teacher")
    else:
        phone_model, phone_ckpt = load_phoneme_aligner(args.phoneme_ckpt, args.device)
        phone_to_id = phone_ckpt["phone_to_id"]
        phone_blank = int(phone_ckpt["blank_id"])
        phone_alpha = float(phone_ckpt["args"].get("alpha", 0.3))
        phone_log_priors = phone_ckpt["log_priors"].float().numpy()
        lexicon = load_lexicon(phone_ckpt["args"].get("lexicon") or None)

    manifest_dir = Path(args.manifest_out).parent
    out_dir = Path(args.out_dir)
    (manifest_dir / out_dir.name).mkdir(parents=True, exist_ok=True)

    kept = 0
    reused = 0
    skipped = {"lexicon": 0, "mapping": 0, "empty": 0}
    gate_means = []
    coverage_p10 = []

    for idx, row in enumerate(rows):
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != bpe_blank]
        if not bpe_ids:
            skipped["empty"] += 1
            continue

        if args.teacher_only:
            rel = Path(out_dir.name) / f"{idx:06d}.pt"
            wav = load_audio(row["audio_filepath"], sample_rate)
            teacher_lp = run_teacher(teacher, wav, args.device)          # (T, V) sharp = WHAT
            # WHERE: gamma from the fine-tuned model if given, else the (de-peaked) teacher itself.
            if where_model is not None:
                where_lp = run_teacher(where_model, wav, args.device)
                where_lp = blank_penalty_logprobs(where_lp, bpe_blank, args.teacher_blank_penalty)
                # align WHERE frame count to teacher frame count if they differ
                if where_lp.shape[0] != teacher_lp.shape[0]:
                    where_lp = resample_time(where_lp, teacher_lp.shape[0])
            else:
                where_lp = blank_penalty_logprobs(teacher_lp, bpe_blank, args.teacher_blank_penalty)
            gamma = ctc_forward_backward(where_lp, bpe_ids, bpe_blank)["token_gamma"]  # (T, N) WHERE
            # WHAT: sharp original-teacher posterior, weighted by that gamma for locality
            avg_ids, avg_probs = teacher_semantic_topk(
                teacher_lp, gamma, bpe_blank, args.top_k, keep_blank=args.keep_blank)
            gates = np.ones(len(bpe_ids), dtype=np.float32)             # teacher-only: no aligner disagreement
            torch.save(
                {
                    "support": torch.tensor(gamma.T, dtype=torch.float16),  # (N, T)
                    "avg_ids": torch.tensor(avg_ids, dtype=torch.int32),
                    "avg_probs": torch.tensor(avg_probs, dtype=torch.float16),
                    "gates": torch.tensor(gates, dtype=torch.float16),
                    "teacher_frames": int(gamma.shape[0]),
                    "num_tokens": int(len(bpe_ids)),
                    "top_k": int(args.top_k),
                    "teacher_blank_penalty": float(args.teacher_blank_penalty),
                    "mode": "teacher_only",
                },
                manifest_dir / rel,
            )
            row["teacher_span_kd_path"] = str(rel)
            kept += 1
            gate_means.append(float(gates.mean()))
            coverage_p10.append(float(np.quantile(gates, 0.1)))
            if kept % args.flush_every == 0:
                write_manifest(args.manifest_out, rows)
                print(f"[prog] kept={kept} processed={idx+1}/{len(rows)}", flush=True)
            continue

        phone_ids, _ = text_to_phone_ids(row["text"], lexicon, phone_to_id, unk_policy=args.unk_policy)
        if not phone_ids:
            skipped["lexicon"] += 1
            continue
        bpe_toks = tokenizer.ids_to_tokens(bpe_ids)
        spans, phone_seq, missing = map_bpe_to_phone_spans(
            row["text"], bpe_toks, lexicon, span_mode=args.span_mode
        )
        if spans is None or not spans:
            skipped["mapping"] += 1
            continue

        rel = Path(out_dir.name) / f"{idx:06d}.pt"
        out_path = manifest_dir / rel
        if args.resume and out_path.exists():
            try:
                cached = torch.load(out_path, map_location="cpu", weights_only=False)
                row["teacher_span_kd_path"] = str(rel)
                kept += 1
                reused += 1
                gate_vals = cached.get("gates")
                if gate_vals is not None:
                    gate_np = gate_vals.float().numpy()
                    gate_means.append(float(gate_np.mean()))
                    coverage_p10.append(float(np.quantile(gate_np, 0.1)))
                if kept % args.flush_every == 0:
                    write_manifest(args.manifest_out, rows)
                    print(
                        f"[prog] kept={kept} reused={reused} processed={idx+1}/{len(rows)}",
                        flush=True,
                    )
                continue
            except Exception as exc:
                print(f"[warn] failed to reuse {out_path}: {exc}; rebuilding", flush=True)

        wav = load_audio(row["audio_filepath"], sample_rate)
        teacher_lp = run_teacher(teacher, wav, args.device)
        teacher_gamma = ctc_forward_backward(teacher_lp, bpe_ids, bpe_blank)["token_gamma"]

        phone_scores = run_phone_model(phone_model, wav, phone_log_priors, phone_alpha, args.device)
        phone_occ = ctc_forward_backward(phone_scores, phone_ids, phone_blank)["token_gamma"]

        support_phone = np.zeros((phone_occ.shape[0], len(bpe_ids)), dtype=np.float64)
        for u, (s, e) in enumerate(spans):
            support_phone[:, u] = phone_occ[:, s:e].sum(axis=1)
        support = resample_time(support_phone, teacher_gamma.shape[0])  # (T, N)

        gt = normalize_cols(teacher_gamma)
        support_peak = support / np.maximum(support.max(axis=0, keepdims=True), 1e-12)
        gates = (gt * support_peak).sum(axis=0).astype(np.float32)

        target_support = support if args.teacher_target_weighting == "gamma_support" else None
        avg_ids, avg_probs = teacher_semantic_topk(
            teacher_lp,
            teacher_gamma,
            bpe_blank,
            args.top_k,
            support=target_support,
        )

        torch.save(
            {
                "support": torch.tensor(support.T, dtype=torch.float16),  # (N, T_teacher)
                "avg_ids": torch.tensor(avg_ids, dtype=torch.int32),
                "avg_probs": torch.tensor(avg_probs, dtype=torch.float16),
                "gates": torch.tensor(gates, dtype=torch.float16),
                "teacher_frames": int(teacher_gamma.shape[0]),
                "num_tokens": int(len(bpe_ids)),
                "top_k": int(args.top_k),
                "span_mode": args.span_mode,
                "teacher_target_weighting": args.teacher_target_weighting,
                "phoneme_ckpt": args.phoneme_ckpt,
            },
            manifest_dir / rel,
        )
        row["teacher_span_kd_path"] = str(rel)
        kept += 1
        gate_means.append(float(gates.mean()))
        coverage_p10.append(float(np.quantile(gates, 0.1)))
        if kept % args.flush_every == 0:
            write_manifest(args.manifest_out, rows)
            print(
                f"[prog] kept={kept} reused={reused} processed={idx+1}/{len(rows)}",
                flush=True,
            )

    write_manifest(args.manifest_out, rows)
    print("\n========== summary ==========")
    print(f"input rows        : {len(rows)}")
    print(f"kept              : {kept}")
    print(f"reused            : {reused}")
    print(f"skipped           : {skipped}")
    if gate_means:
        print(f"gate mean avg     : {np.mean(gate_means):.4f}")
        print(f"gate p10 avg      : {np.mean(coverage_p10):.4f}")
    print(f"written           : {args.manifest_out}")


if __name__ == "__main__":
    main()
