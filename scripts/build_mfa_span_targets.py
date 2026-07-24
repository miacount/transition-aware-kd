#!/usr/bin/env python3
"""Build span-KD targets whose SUPPORT (WHERE) comes from MFA forced alignments
instead of the teacher's own de-peaked forward-backward gamma.

Purpose (D2 ablation, strong-aligner version): everything else in the span-KD
pipeline is unchanged — same .pt format, same training loss, same WHAT source
(sharp teacher posterior) — only the coordinate system that defines "where each
token lives" is swapped from teacher-self to an external aligner (MFA).

Per token u:
  WHERE  support(u, t) = overlap of frame t with u's MFA-derived interval.
         MFA gives word intervals; multi-BPE words are subdivided among their
         BPE tokens proportionally to grapheme length (no FB anywhere).
  WHAT   q_u = top-k of  sum_t w(t,u) * p_teacher(t, .)  with blank dropped,
         where w = support normalized over time — i.e. the teacher's sharp
         posterior averaged over the SAME support the student will be averaged
         over (exactly parallel to the original method's structure).

Utterances whose transcript does not match the MFA word sequence are skipped
(no teacher_span_kd_path -> dropped by the span_kd dataloader); counts printed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))


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
    assert sr == sample_rate, f"sample rate {sr} != {sample_rate}"
    return wav


@torch.no_grad()
def run_teacher(model, wav_np, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t, input_signal_length=wav_len)
    frames = int(enc_len[0].item())
    return log_probs[0, :frames].detach().cpu().float().numpy()


def group_bpe_words(tokens_str):
    groups = []
    for i, tok in enumerate(tokens_str):
        if tok.startswith("▁") or i == 0:
            groups.append([i, i + 1])
        else:
            groups[-1][1] = i + 1
    return groups


def interval_support(start_s, end_s, fd, T, pad_frames):
    """support[t] = seconds of overlap between frame t and [start-pad, end+pad)."""
    lo = max(0.0, start_s - pad_frames * fd)
    hi = end_s + pad_frames * fd
    t0 = max(0, int(np.floor(lo / fd)))
    t1 = min(T, int(np.ceil(hi / fd)))
    sup = np.zeros(T, dtype=np.float64)
    for t in range(t0, t1):
        sup[t] = max(0.0, min(hi, (t + 1) * fd) - max(lo, t * fd))
    if sup.sum() <= 0.0:  # degenerate (interval shorter than grid / clipped)
        mid = int(np.clip(0.5 * (start_s + end_s) / fd, 0, T - 1))
        sup[mid] = 1.0
    return sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-in", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--alignments", required=True,
                    help="json: utt_id -> {words: [{word,start,end}, ...]}")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--out-dir", default="data/span_kd_mfa",
                    help="targets stored under <manifest_out dir>/<basename(out-dir)>")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--pad-frames", type=float, default=0.0,
                    help="extend each token interval by this many frames on each side")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[:args.limit]
    with open(args.alignments) as f:
        mfa = json.load(f)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    manifest_dir = Path(args.manifest_out).parent
    out_dir = Path(args.out_dir)
    (manifest_dir / out_dir.name).mkdir(parents=True, exist_ok=True)

    kept = reused = 0
    n_unk = 0
    skipped = {"no_mfa": 0, "word_mismatch": 0, "bpe_group": 0, "empty": 0}

    for idx, row in enumerate(rows):
        utt_id = Path(row["audio_filepath"]).stem
        if utt_id not in mfa:
            skipped["no_mfa"] += 1
            continue
        words_mfa = mfa[utt_id]["words"]
        words_txt = row["text"].split()
        # MFA labels dictionary-OOV words "<unk>" but still aligns their interval;
        # accept them as wildcards instead of discarding the utterance.
        if len(words_mfa) != len(words_txt) or any(
                w["word"].lower() not in (t, "<unk>")
                for w, t in zip(words_mfa, words_txt)):
            skipped["word_mismatch"] += 1
            continue
        n_unk += sum(1 for w in words_mfa if w["word"].lower() == "<unk>")

        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != blank]
        if not bpe_ids:
            skipped["empty"] += 1
            continue
        toks = tokenizer.ids_to_tokens(bpe_ids)
        groups = group_bpe_words(toks)
        if len(groups) != len(words_txt):
            skipped["bpe_group"] += 1
            continue

        rel = Path(out_dir.name) / f"{idx:06d}.pt"
        out_path = manifest_dir / rel
        if args.resume and out_path.exists():
            row["teacher_span_kd_path"] = str(rel)
            kept += 1
            reused += 1
            continue

        wav = load_audio(row["audio_filepath"], sample_rate)
        teacher_lp = run_teacher(teacher, wav, args.device)   # (T, V) sharp
        T = teacher_lp.shape[0]
        fd = (len(wav) / sample_rate) / T
        N = len(bpe_ids)

        # WHERE: MFA word interval, split among the word's BPE tokens by char length
        support = np.zeros((N, T), dtype=np.float64)
        for wi, (a, b) in enumerate(groups):
            ws, we = float(words_mfa[wi]["start"]), float(words_mfa[wi]["end"])
            lens = np.array([max(len(t.lstrip("▁")), 1) for t in toks[a:b]], dtype=np.float64)
            edges = ws + (we - ws) * np.concatenate([[0.0], np.cumsum(lens) / lens.sum()])
            for j, u in enumerate(range(a, b)):
                support[u] = interval_support(edges[j], edges[j + 1], fd, T, args.pad_frames)

        # WHAT: sharp teacher posterior averaged over the SAME support, blank dropped
        probs = np.exp(teacher_lp)
        w = support / np.maximum(support.sum(axis=1, keepdims=True), 1e-12)
        q = w @ probs                                          # (N, V)
        q[:, blank] = 0.0
        q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        k = min(args.top_k, q.shape[1])
        idx_k = np.argpartition(q, -k, axis=1)[:, -k:]
        order = np.argsort(np.take_along_axis(q, idx_k, 1), axis=1)[:, ::-1]
        avg_ids = np.take_along_axis(idx_k, order, 1)
        avg_probs = np.take_along_axis(q, avg_ids, 1)
        avg_probs = avg_probs / np.maximum(avg_probs.sum(axis=1, keepdims=True), 1e-12)

        torch.save(
            {
                "support": torch.tensor(support, dtype=torch.float16),      # (N, T)
                "avg_ids": torch.tensor(avg_ids.astype(np.int32)),
                "avg_probs": torch.tensor(avg_probs.astype(np.float16)),
                "gates": torch.ones(N, dtype=torch.float16),
                "teacher_frames": int(T),
                "num_tokens": int(N),
                "top_k": int(args.top_k),
                "pad_frames": float(args.pad_frames),
                "mode": "mfa_support",
                "subdiv": "char_proportional",
            },
            out_path,
        )
        row["teacher_span_kd_path"] = str(rel)
        kept += 1
        if kept % args.flush_every == 0:
            write_manifest(args.manifest_out, rows)
            print(f"[prog] kept={kept} processed={idx+1}/{len(rows)}", flush=True)

    write_manifest(args.manifest_out, rows)
    print("\n========== summary ==========")
    print(f"input rows : {len(rows)}")
    print(f"kept       : {kept} (reused {reused})")
    print(f"unk words  : {n_unk} (interval kept, label wildcarded)")
    print(f"skipped    : {skipped}")
    print(f"written    : {args.manifest_out}")


if __name__ == "__main__":
    main()
