#!/usr/bin/env python
"""Build teacher KD manifests.

Modes:
  transition : add teacher_target from teacher greedy CTC path collapse
  frame_topk : add teacher_frame_kd_path with per-frame top-k posterior .pt files
  frame_dense: add teacher_frame_kd_path with the full per-frame posterior
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def collapse_repeats(ids):
    out = []
    prev = None
    for x in ids:
        if x != prev:
            out.append(int(x))
            prev = x
    return out


FLOOR = -1e9


def viterbi_forced_align(log_probs_np, tokens, blank_id):
    """CTC forced alignment. tokens: list of int (no blanks). Returns per-frame token id array."""
    ext = []
    for t in tokens:
        ext.append(blank_id)
        ext.append(t)
    ext.append(blank_id)
    S, T = len(ext), log_probs_np.shape[0]
    alpha = np.full((T, S), FLOOR)
    bp = np.zeros((T, S), dtype=np.int32)
    alpha[0, 0] = log_probs_np[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs_np[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, src = alpha[t - 1, s], s
            if s > 0 and alpha[t - 1, s - 1] > best:
                best, src = alpha[t - 1, s - 1], s - 1
            if s > 1 and ext[s] != blank_id and ext[s] != ext[s - 2] and alpha[t - 1, s - 2] > best:
                best, src = alpha[t - 1, s - 2], s - 2
            alpha[t, s] = best + log_probs_np[t, ext[s]]
            bp[t, s] = src
    s = S - 1 if alpha[T - 1, S - 1] > alpha[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = s
    for t in range(T - 2, -1, -1):
        s = bp[t + 1, s]
        path[t] = s
    return np.array([ext[s] for s in path], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["transition", "frame_topk", "frame_dense", "token_avg"],
                    required=True)
    ap.add_argument("--manifest_in", required=True)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--resume", action="store_true",
                    help="reuse valid per-utterance target files and continue an interrupted build")
    ap.add_argument("--flush_every", type=int, default=100,
                    help="write a partial manifest after this many newly generated targets")
    ap.add_argument("--confidence_threshold", type=float, default=0.0,
                    help="token_avg: skip segment if avg_p[token] < threshold (0=disabled)")
    ap.add_argument("--min_seg_len", type=int, default=1,
                    help="token_avg: skip segment if non-blank frame count < min_seg_len")
    args = ap.parse_args()

    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    from torch.nn.utils.rnn import pad_sequence

    teacher_name = "stt_en_conformer_ctc_small" if args.teacher == "stt_en_conformer_small" else args.teacher
    print(f"[load] {teacher_name}")
    if teacher_name.endswith(".nemo"):
        teacher = nemo_asr.models.EncDecCTCModelBPE.restore_from(teacher_name)
    else:
        teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(teacher_name)
    teacher = teacher.to(args.device).eval()
    teacher.freeze()
    sample_rate = teacher.cfg.preprocessor.sample_rate
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    print(f"[blank_id] {blank_id}")

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "data/frame_kd" if args.mode in ("frame_topk", "frame_dense") else "data/token_avg_kd"
    )
    manifest_dir = Path(args.manifest_out).parent
    if args.mode in ("frame_topk", "frame_dense", "token_avg"):
        (manifest_dir / out_dir.name).mkdir(parents=True, exist_ok=True)

    def load_audio(path):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        return torch.from_numpy(wav)

    t_lengths = []
    target_lengths = []
    topk_masses = []
    n_segs_total = 0
    n_filtered_len = 0
    n_filtered_conf = 0
    reused = 0
    generated = 0

    # Expensive dense targets are restartable. Build only missing/corrupt files;
    # rows with a valid cache are still written to the output manifest.
    process_indices = list(range(len(rows)))
    if args.mode in ("frame_topk", "frame_dense", "token_avg") and args.resume:
        process_indices = []
        for idx, row in enumerate(rows):
            rel = Path(out_dir.name) / f"{idx:06d}.pt"
            out_path = manifest_dir / rel
            if out_path.exists():
                try:
                    cached = torch.load(out_path, map_location="cpu", weights_only=False)
                    if args.mode == "frame_dense":
                        valid = "dense_probs" in cached
                        key = "teacher_frame_kd_path"
                    elif args.mode == "frame_topk":
                        valid = "ids" in cached and "probs" in cached
                        key = "teacher_frame_kd_path"
                    else:
                        valid = "avg_ids" in cached and "avg_probs" in cached
                        key = "teacher_token_avg_path"
                    if valid:
                        row[key] = str(rel)
                        reused += 1
                        continue
                except Exception as exc:
                    print(f"[warn] failed to reuse {out_path}: {exc}", flush=True)
            process_indices.append(idx)


    with torch.no_grad():
        for start in range(0, len(process_indices), args.batch_size):
            batch_indices = process_indices[start : start + args.batch_size]
            batch = [rows[idx] for idx in batch_indices]
            sigs = [load_audio(row["audio_filepath"]) for row in batch]
            lens = torch.tensor([x.numel() for x in sigs], dtype=torch.long, device=args.device)
            padded = pad_sequence(sigs, batch_first=True).to(args.device)
            log_probs, enc_len, greedy = teacher.forward(input_signal=padded, input_signal_length=lens)
            greedy = greedy.detach().cpu()
            enc_len_cpu = enc_len.detach().cpu()

            for i, row in enumerate(batch):
                global_idx = batch_indices[i]
                frames = int(enc_len_cpu[i].item())
                t_lengths.append(frames)

                if args.mode == "transition":
                    target = collapse_repeats(greedy[i, :frames].tolist())
                    row["teacher_target"] = target
                    target_lengths.append(len(target))
                elif args.mode == "frame_topk":
                    lp = log_probs[i, :frames].float()
                    if args.temperature != 1.0:
                        lp = torch.log_softmax(lp / args.temperature, dim=-1)
                    vals, ids = torch.topk(lp, k=min(args.top_k, lp.shape[-1]), dim=-1)
                    probs = vals.exp()
                    topk_masses.append(float(probs.sum(dim=-1).mean().detach().cpu()))
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    rel = Path(out_dir.name) / f"{global_idx:06d}.pt"
                    torch.save(
                        {
                            "ids": ids.detach().cpu().to(torch.int32),
                            "probs": probs.detach().cpu().to(torch.float16),
                            "frames": frames,
                            "top_k": int(ids.shape[-1]),
                            "temperature": float(args.temperature),
                        },
                        manifest_dir / rel,
                    )
                    row["teacher_frame_kd_path"] = str(rel)
                elif args.mode == "frame_dense":
                    lp = log_probs[i, :frames].float()
                    if args.temperature != 1.0:
                        lp = torch.log_softmax(lp / args.temperature, dim=-1)
                    rel = Path(out_dir.name) / f"{global_idx:06d}.pt"
                    torch.save(
                        {
                            "dense_probs": lp.exp().detach().cpu().to(torch.bfloat16),
                            "frames": frames,
                            "vocab_size": int(lp.shape[-1]),
                            "temperature": float(args.temperature),
                            "mode": "frame_dense",
                        },
                        manifest_dir / rel,
                    )
                    row["teacher_frame_kd_path"] = str(rel)

                else:  # token_avg
                    lp = log_probs[i, :frames].float()
                    prob = lp.exp().cpu().numpy()  # (T, K)
                    lp_np = lp.cpu().numpy()

                    # GT transcript → token sequence (GT-Viterbi alignment)
                    # Using GT anchors teacher posteriors to correct token positions,
                    # avoiding pseudo-labeling teacher errors.
                    tokens = teacher.tokenizer.text_to_ids(row["text"])
                    tokens = [t for t in tokens if t != blank_id]

                    if not tokens:
                        # silent/empty utterance — skip KD target
                        continue

                    frame_tokens = viterbi_forced_align(lp_np, tokens, blank_id)

                    # collect non-blank segments
                    seg_starts, seg_ends, avg_ids_list, avg_probs_list = [], [], [], []
                    trans_left_ids_list, trans_left_probs_list, trans_left_valid_list = [], [], []
                    trans_right_ids_list, trans_right_probs_list, trans_right_valid_list = [], [], []
                    j = 0
                    while j < len(frame_tokens):
                        tok = int(frame_tokens[j])
                        k = j
                        while k < len(frame_tokens) and int(frame_tokens[k]) == tok:
                            k += 1
                        if tok != blank_id:
                            n_segs_total += 1
                            seg_len = k - j
                            if seg_len < args.min_seg_len:
                                n_filtered_len += 1
                                j = k
                                continue
                            seg_prob = prob[j:k]          # (seg_len, K)
                            avg_p = seg_prob.mean(axis=0)  # (K,)
                            if args.confidence_threshold > 0.0 and avg_p[tok] < args.confidence_threshold:
                                n_filtered_conf += 1
                                j = k
                                continue
                            top_k = min(args.top_k, len(avg_p))
                            top_idx = np.argpartition(avg_p, -top_k)[-top_k:]
                            top_idx = top_idx[np.argsort(avg_p[top_idx])[::-1]]
                            top_probs = avg_p[top_idx]
                            top_probs = top_probs / top_probs.sum()
                            seg_starts.append(j)
                            seg_ends.append(k)
                            avg_ids_list.append(top_idx.astype(np.int32))
                            avg_probs_list.append(top_probs.astype(np.float32))

                            # left transition: frame j-1 (blank frame before segment)
                            if j > 0 and int(frame_tokens[j - 1]) == blank_id:
                                lp = prob[j - 1]
                                li = np.argpartition(lp, -top_k)[-top_k:]
                                li = li[np.argsort(lp[li])[::-1]]
                                lv = lp[li]; lv = lv / lv.sum()
                                trans_left_ids_list.append(li.astype(np.int32))
                                trans_left_probs_list.append(lv.astype(np.float32))
                                trans_left_valid_list.append(True)
                            else:
                                trans_left_ids_list.append(np.zeros(top_k, dtype=np.int32))
                                trans_left_probs_list.append(np.zeros(top_k, dtype=np.float32))
                                trans_left_valid_list.append(False)

                            # right transition: frame k (blank frame after segment)
                            if k < frames and int(frame_tokens[k]) == blank_id:
                                rp = prob[k]
                                ri = np.argpartition(rp, -top_k)[-top_k:]
                                ri = ri[np.argsort(rp[ri])[::-1]]
                                rv = rp[ri]; rv = rv / rv.sum()
                                trans_right_ids_list.append(ri.astype(np.int32))
                                trans_right_probs_list.append(rv.astype(np.float32))
                                trans_right_valid_list.append(True)
                            else:
                                trans_right_ids_list.append(np.zeros(top_k, dtype=np.int32))
                                trans_right_probs_list.append(np.zeros(top_k, dtype=np.float32))
                                trans_right_valid_list.append(False)
                        j = k

                    if not seg_starts:
                        # all segments filtered — skip KD target for this utterance
                        continue
                    rel = Path(out_dir.name) / f"{global_idx:06d}.pt"
                    torch.save(
                        {
                            "seg_starts": torch.tensor(seg_starts, dtype=torch.int32),
                            "seg_ends":   torch.tensor(seg_ends,   dtype=torch.int32),
                            "avg_ids":    torch.tensor(np.stack(avg_ids_list),   dtype=torch.int32),
                            "avg_probs":  torch.tensor(np.stack(avg_probs_list), dtype=torch.float16),
                            "trans_left_ids":   torch.tensor(np.stack(trans_left_ids_list),   dtype=torch.int32),
                            "trans_left_probs": torch.tensor(np.stack(trans_left_probs_list), dtype=torch.float16),
                            "trans_left_valid": torch.tensor(trans_left_valid_list,            dtype=torch.bool),
                            "trans_right_ids":   torch.tensor(np.stack(trans_right_ids_list),   dtype=torch.int32),
                            "trans_right_probs": torch.tensor(np.stack(trans_right_probs_list), dtype=torch.float16),
                            "trans_right_valid": torch.tensor(trans_right_valid_list,            dtype=torch.bool),
                            "teacher_frames": frames,
                        },
                        manifest_dir / rel,
                    )
                    row["teacher_token_avg_path"] = str(rel)
                    target_lengths.append(len(seg_starts))
                generated += 1
                if generated % args.flush_every == 0:
                    write_manifest(args.manifest_out, rows)


            print(
                f"[prog] generated={min(start + args.batch_size, len(process_indices))}/"
                f"{len(process_indices)} reused={reused} total={len(rows)}",
                flush=True,
            )

    write_manifest(args.manifest_out, rows)
    print("\n========== summary ==========")
    print(f"mode             : {args.mode}")
    print(f"utterances       : {len(rows)}")
    print(f"generated/reused : {generated}/{reused}")
    if t_lengths:
        print(f"teacher T min/max: {min(t_lengths)}/{max(t_lengths)}")
    if target_lengths:
        print(f"target M min/max : {min(target_lengths)}/{max(target_lengths)}")
    if topk_masses:
        print(f"top-k/temp       : {args.top_k}/{args.temperature}")
        print(f"top-k mass mean  : {sum(topk_masses) / len(topk_masses):.4f}")
    if args.mode == "token_avg" and n_segs_total > 0:
        n_kept = n_segs_total - n_filtered_len - n_filtered_conf
        print(f"segments total   : {n_segs_total}")
        print(f"filtered min_len : {n_filtered_len} ({100*n_filtered_len/n_segs_total:.1f}%)")
        print(f"filtered conf    : {n_filtered_conf} ({100*n_filtered_conf/n_segs_total:.1f}%)")
        print(f"segments kept    : {n_kept} ({100*n_kept/n_segs_total:.1f}%)")
        print(f"conf_threshold   : {args.confidence_threshold}")
        print(f"min_seg_len      : {args.min_seg_len}")
    print(f"written          : {args.manifest_out}")


if __name__ == "__main__":
    main()
