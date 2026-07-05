#!/usr/bin/env python
"""Frame-wise CTC argmax path mismatch diagnostic."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel  # noqa: E402
from nemo.collections.asr.models import EncDecCTCModelBPE  # noqa: E402


def read_rows(path, limit):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_audio(path, sample_rate):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return torch.from_numpy(wav)


def load_student(config, ckpt, device):
    cfg = OmegaConf.load(config)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if model_cfg.get("test_ds") is not None:
            del model_cfg.test_ds
    model = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def argmax_path(model, wav, device):
    wav = wav.unsqueeze(0).to(device)
    wav_len = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav, input_signal_length=wav_len)
    enc_len = int(enc_len[0].item())
    return log_probs[0, :enc_len].argmax(dim=-1).detach().cpu().numpy()


def map_to_len(path, length):
    if len(path) == length:
        return path
    pos = (np.arange(length, dtype=np.float64) + 0.5) / length
    idx = np.floor(pos * len(path)).astype(np.int64)
    return path[np.clip(idx, 0, len(path) - 1)]


def collapse(path):
    out = []
    prev = None
    for x in path:
        x = int(x)
        if x != prev:
            out.append(x)
            prev = x
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return prev[-1]


def safe(num, den):
    return float(num) / float(den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--csv_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    teacher = EncDecCTCModelBPE.from_pretrained(args.teacher).to(device).eval()
    student = load_student(args.config, args.ckpt, device)
    blank = teacher.decoder.num_classes_with_blank - 1

    totals = {k: 0 for k in [
        "frames", "mismatch", "blank_status", "t_nonblank", "t_blank",
        "t_nb_s_blank", "t_blank_s_nb", "both_nonblank", "both_nb_diff",
        "trans_edits", "trans_ref",
    ]}
    per_utt = []
    for idx, row in enumerate(read_rows(args.manifest, args.limit)):
        wav = load_audio(row["audio_filepath"], args.sample_rate)
        t = argmax_path(teacher, wav, device)
        s = map_to_len(argmax_path(student, wav, device), len(t))
        t_blank = t == blank
        s_blank = s == blank
        mismatch = t != s
        both_nb = (~t_blank) & (~s_blank)
        both_nb_diff = both_nb & mismatch
        t_nb_s_blank = (~t_blank) & s_blank
        t_blank_s_nb = t_blank & (~s_blank)
        ted = edit_distance(collapse(t), collapse(s))

        totals["frames"] += len(t)
        totals["mismatch"] += int(mismatch.sum())
        totals["blank_status"] += int((t_blank != s_blank).sum())
        totals["t_nonblank"] += int((~t_blank).sum())
        totals["t_blank"] += int(t_blank.sum())
        totals["t_nb_s_blank"] += int(t_nb_s_blank.sum())
        totals["t_blank_s_nb"] += int(t_blank_s_nb.sum())
        totals["both_nonblank"] += int(both_nb.sum())
        totals["both_nb_diff"] += int(both_nb_diff.sum())
        totals["trans_edits"] += ted
        totals["trans_ref"] += len(collapse(t))
        per_utt.append({
            "idx": idx,
            "audio_filepath": row["audio_filepath"],
            "frames": len(t),
            "frame_mismatch_rate": safe(int(mismatch.sum()), len(t)),
            "blank_nonblank_mismatch_rate": safe(int((t_blank != s_blank).sum()), len(t)),
            "transition_edit_rate": safe(ted, len(collapse(t))),
        })

    print(f"utterances                    : {len(per_utt)}")
    print(f"frames compared               : {totals['frames']}")
    print(f"frame mismatch rate           : {safe(totals['mismatch'], totals['frames']) * 100:.2f}%")
    print(f"blank/nonblank mismatch rate  : {safe(totals['blank_status'], totals['frames']) * 100:.2f}%")
    print(f"teacher nonblank->student blank: {safe(totals['t_nb_s_blank'], totals['t_nonblank']) * 100:.2f}%")
    print(f"teacher blank->student nonblank: {safe(totals['t_blank_s_nb'], totals['t_blank']) * 100:.2f}%")
    print(f"both nonblank token diff rate : {safe(totals['both_nb_diff'], totals['both_nonblank']) * 100:.2f}%")
    nb_token_err = totals['t_nb_s_blank'] + totals['both_nb_diff']
    print(f"NB spike error rate (given teacher NB): {safe(nb_token_err, totals['t_nonblank']) * 100:.2f}%")
    print(f"transition edit rate          : {safe(totals['trans_edits'], totals['trans_ref']) * 100:.2f}%")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if per_utt:
            with out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_utt[0].keys()))
                writer.writeheader()
                writer.writerows(per_utt)
            print(f"written                       : {out}")
        else:
            print(f"[warn] no utterances processed — csv not written")


if __name__ == "__main__":
    main()
