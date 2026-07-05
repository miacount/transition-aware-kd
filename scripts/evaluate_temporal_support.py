#!/usr/bin/env python3
"""
Evaluate phoneme soft aligner and phoneme-derived BPE temporal support.

This is a pre-KD sanity check. It answers:
  1. Is the phoneme aligner less blank-dominated and reasonably monotonic?
  2. Does phoneme-derived BPE support cover teacher BPE CTC cores?
  3. Does the teacher semantic distribution for each token occurrence still
     point to the transcript BPE token?

Run:
  python scripts/evaluate_temporal_support.py \
    --phoneme-ckpt nemo_experiments/phoneme_soft_aligner/epoch_010.pt \
    --manifest data/dev_clean.json \
    --limit 256
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
from train_soft_aligner import SoftAligner, _wav_to_mel  # noqa: E402
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402
from visualize_temporal_support_kd import (  # noqa: E402
    load_phoneme_aligner,
    map_bpe_to_phone_spans,
    normalize_cols,
    resample_time,
)


def read_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_audio(path, sample_rate=16000):
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
    scores = raw_lp - alpha * log_priors.reshape(1, -1)
    scores_norm = scores - scores.max(axis=1, keepdims=True)
    adj_probs = np.exp(scores_norm)
    adj_probs /= np.maximum(adj_probs.sum(axis=1, keepdims=True), 1e-12)
    return raw_lp, scores, adj_probs


def weighted_mean_time(mat):
    # mat: (T, U), nonnegative.
    T = mat.shape[0]
    t = np.arange(T, dtype=np.float64)[:, None]
    denom = np.maximum(mat.sum(axis=0), 1e-12)
    return (mat * t).sum(axis=0) / denom


def monotonicity_violations(mat):
    centers = weighted_mean_time(mat)
    if len(centers) <= 1:
        return 0.0
    return float(np.mean(np.diff(centers) < -0.5))


def token_effective_width(mat):
    """Average effective support width in frames, 1/sum p_t^2 after col norm."""
    col = normalize_cols(mat)
    eff = 1.0 / np.maximum((col * col).sum(axis=0), 1e-12)
    return eff


def entropy_rows(probs):
    return -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.1)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-ckpt", default="nemo_experiments/phoneme_soft_aligner/epoch_010.pt")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--span-mode", choices=["proportional", "word"], default="proportional")
    ap.add_argument("--out", default="analysis/temporal_support_kd/eval_epoch010.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    bpe_blank = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    phone_model, phone_ckpt = load_phoneme_aligner(args.phoneme_ckpt, args.device)
    phone_to_id = phone_ckpt["phone_to_id"]
    phone_blank = int(phone_ckpt["blank_id"])
    phone_alpha = float(phone_ckpt["args"].get("alpha", 0.3))
    phone_log_priors = phone_ckpt["log_priors"].float().numpy()
    lexicon = load_lexicon(phone_ckpt["args"].get("lexicon") or None)

    rows = read_rows(args.manifest)
    utt_stats = []
    token_stats = []
    skipped = {"lexicon": 0, "mapping": 0, "empty": 0}

    for row in rows:
        if len(utt_stats) >= args.limit:
            break
        phone_ids, missing = text_to_phone_ids(row["text"], lexicon, phone_to_id, unk_policy="skip")
        if not phone_ids:
            skipped["lexicon"] += 1
            continue
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != bpe_blank]
        if not bpe_ids:
            skipped["empty"] += 1
            continue
        bpe_toks = tokenizer.ids_to_tokens(bpe_ids)
        spans, phone_seq, missing = map_bpe_to_phone_spans(
            row["text"], bpe_toks, lexicon, span_mode=args.span_mode
        )
        if missing or not spans:
            skipped["mapping"] += 1
            continue

        wav = load_audio(row["audio_filepath"], sample_rate)
        teacher_lp = run_teacher(teacher, wav, args.device)
        teacher_fb = ctc_forward_backward(teacher_lp, bpe_ids, bpe_blank)
        teacher_gamma = teacher_fb["token_gamma"]

        raw_phone_lp, phone_scores, phone_probs = run_phone_model(
            phone_model, wav, phone_log_priors, phone_alpha, args.device
        )
        phone_fb = ctc_forward_backward(phone_scores, phone_ids, phone_blank)
        phone_occ = phone_fb["token_gamma"]

        bpe_mask_phone = np.zeros((phone_occ.shape[0], len(bpe_ids)), dtype=np.float64)
        for u, (s, e) in enumerate(spans):
            bpe_mask_phone[:, u] = phone_occ[:, s:e].sum(axis=1)
        bpe_mask = resample_time(bpe_mask_phone, teacher_gamma.shape[0])

        gt = normalize_cols(teacher_gamma)
        ga = normalize_cols(bpe_mask)
        dot_gates = (gt * ga).sum(axis=0)
        mask_peak = bpe_mask / np.maximum(bpe_mask.max(axis=0, keepdims=True), 1e-12)
        coverage_gates = (gt * mask_peak).sum(axis=0)

        probs_t = np.exp(teacher_lp)
        raw_probs_phone = np.exp(raw_phone_lp)
        q_top_match = []
        q_top_prob = []
        q_entropy = []
        eff_teacher = token_effective_width(teacher_gamma)
        eff_mask = token_effective_width(bpe_mask)

        for u, tok in enumerate(bpe_ids):
            q = (teacher_gamma[:, u:u + 1] * probs_t).sum(axis=0)
            q[bpe_blank] = 0.0
            q_sum = max(float(q.sum()), 1e-12)
            q = q / q_sum
            q_top = int(q.argmax())
            q_top_match.append(q_top == int(tok))
            q_top_prob.append(float(q[int(tok)]))
            q_entropy.append(float(-(q * np.log(np.maximum(q, 1e-12))).sum()))
            token_stats.append({
                "dot_gate": float(dot_gates[u]),
                "coverage_gate": float(coverage_gates[u]),
                "teacher_top1_match": bool(q_top == int(tok)),
                "teacher_gt_prob": float(q[int(tok)]),
                "teacher_q_entropy": q_entropy[-1],
                "teacher_width": float(eff_teacher[u]),
                "support_width": float(eff_mask[u]),
            })

        phone_vit = np.argmax(phone_probs, axis=1)
        utt_stats.append({
            "audio_filepath": row["audio_filepath"],
            "text": row["text"],
            "frames_teacher": int(teacher_gamma.shape[0]),
            "frames_phone": int(phone_occ.shape[0]),
            "bpe_tokens": int(len(bpe_ids)),
            "phones": int(len(phone_ids)),
            "raw_phone_blank": float(raw_probs_phone[:, phone_blank].mean()),
            "adj_phone_blank": float(phone_probs[:, phone_blank].mean()),
            "phone_entropy": float(entropy_rows(phone_probs).mean()),
            "phone_argmax_nonblank_rate": float(np.mean(phone_vit != phone_blank)),
            "phone_occ_monotonic_viol": monotonicity_violations(phone_occ),
            "teacher_gamma_monotonic_viol": monotonicity_violations(teacher_gamma),
            "bpe_support_monotonic_viol": monotonicity_violations(bpe_mask),
            "dot_gate_mean": float(dot_gates.mean()),
            "coverage_gate_mean": float(coverage_gates.mean()),
            "teacher_top1_match_rate": float(np.mean(q_top_match)),
            "teacher_gt_prob_mean": float(np.mean(q_top_prob)),
            "teacher_q_entropy_mean": float(np.mean(q_entropy)),
            "teacher_width_mean": float(np.mean(eff_teacher)),
            "support_width_mean": float(np.mean(eff_mask)),
        })

    def collect(key, source):
        return [x[key] for x in source]

    summary = {
        "phoneme_ckpt": args.phoneme_ckpt,
        "manifest": args.manifest,
        "span_mode": args.span_mode,
        "utterances": len(utt_stats),
        "tokens": len(token_stats),
        "skipped": skipped,
        "utterance_metrics": {
            key: summarize(collect(key, utt_stats))
            for key in [
                "raw_phone_blank", "adj_phone_blank", "phone_entropy",
                "phone_argmax_nonblank_rate", "phone_occ_monotonic_viol",
                "teacher_gamma_monotonic_viol", "bpe_support_monotonic_viol",
                "dot_gate_mean", "coverage_gate_mean",
                "teacher_top1_match_rate", "teacher_gt_prob_mean",
                "teacher_q_entropy_mean",
                "teacher_width_mean", "support_width_mean",
            ]
        },
        "token_metrics": {
            key: summarize(collect(key, token_stats))
            for key in [
                "dot_gate", "coverage_gate", "teacher_gt_prob",
                "teacher_q_entropy", "teacher_width", "support_width",
            ]
        },
        "teacher_top1_match_token_rate": float(np.mean([x["teacher_top1_match"] for x in token_stats])) if token_stats else 0.0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
