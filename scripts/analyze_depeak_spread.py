#!/usr/bin/env python
"""Decompose a student's "over-firing" frames — where the teacher argmax is blank
but the student emits a non-blank — by distance to the nearest teacher non-blank
spike, to tell intended span-spreading (shoulder frames beside a real spike) from
spurious emission (tokens in teacher-blank regions with no spike nearby).

For each over-firing frame it reports:
  - shoulder (dist<=SHOULDER): beside a teacher spike. Split by whether the
    student's token EQUALS that nearest spike's token (clean spread) or differs.
  - isolated (dist>SHOULDER): no teacher spike nearby (potential hallucination).

Same teacher/student argmax extraction as diagnose_ctc_mismatch.py (sub4: equal
frame rate, 1:1 frame comparison after length matching).
"""
import argparse, json, os, sys
import numpy as np, soundfile as sf, torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel
from nemo.collections.asr.models import EncDecCTCModelBPE


def load_audio(path, sr):
    wav, s = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if s != sr:
        import librosa; wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    return torch.from_numpy(wav)


def load_student(config, ckpt, device):
    cfg = OmegaConf.load(config); mc = cfg.model.copy()
    with open_dict(mc):
        mc.log_prediction = False
        if mc.get("test_ds") is not None: del mc.test_ds
    m = TransitionKDModel(cfg=mc, trainer=None)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(st.get("state_dict", st), strict=False)
    return m.to(device).eval()


@torch.no_grad()
def argmax_path(model, wav, device):
    wav = wav.unsqueeze(0).to(device)
    wl = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)
    lp, el, _ = model.forward(input_signal=wav, input_signal_length=wl)
    el = int(el[0].item())
    return lp[0, :el].argmax(dim=-1).detach().cpu().numpy()


def map_to_len(path, length):
    if len(path) == length: return path
    pos = (np.arange(length, dtype=np.float64) + 0.5) / length
    idx = np.floor(pos * len(path)).astype(np.int64)
    return path[np.clip(idx, 0, len(path) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/tedlium_test.json")
    ap.add_argument("--config", default="configs/student_base_ted.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--shoulder", type=int, default=2,
                    help="max frame distance to a teacher spike to count as shoulder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    teacher = EncDecCTCModelBPE.from_pretrained(args.teacher).to(dev).eval()
    student = load_student(args.config, args.ckpt, dev)
    blank = teacher.decoder.num_classes_with_blank - 1

    C = {"overfire": 0, "shoulder": 0, "shoulder_same": 0, "shoulder_diff": 0,
         "isolated": 0, "t_blank": 0, "t_nonblank": 0, "frames": 0}
    dist_hist = {}
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()][:args.limit]
    for row in rows:
        wav = load_audio(row["audio_filepath"], args.sample_rate)
        tp = argmax_path(teacher, wav, dev)
        sp = argmax_path(student, wav, dev)
        sp = map_to_len(sp, len(tp))
        t_nb = np.where(tp != blank)[0]           # teacher spike frame indices
        C["frames"] += len(tp)
        C["t_blank"] += int((tp == blank).sum())
        C["t_nonblank"] += len(t_nb)
        if len(t_nb) == 0:
            # whole utt teacher-blank: every student non-blank is isolated
            iso = np.where(sp != blank)[0]
            C["overfire"] += len(iso); C["isolated"] += len(iso)
            continue
        over = np.where((tp == blank) & (sp != blank))[0]
        C["overfire"] += len(over)
        for f in over:
            j = int(np.searchsorted(t_nb, f))
            cands = []
            if j < len(t_nb): cands.append(t_nb[j])
            if j > 0: cands.append(t_nb[j - 1])
            nearest = min(cands, key=lambda x: abs(x - f))
            d = abs(int(nearest) - int(f))
            dist_hist[d] = dist_hist.get(d, 0) + 1
            if d <= args.shoulder:
                C["shoulder"] += 1
                if int(sp[f]) == int(tp[nearest]): C["shoulder_same"] += 1
                else: C["shoulder_diff"] += 1
            else:
                C["isolated"] += 1

    def pct(a, b): return 100.0 * a / b if b else 0.0
    print(f"ckpt: {args.ckpt}")
    print(f"utts {len(rows)} | frames {C['frames']} | teacher blank {pct(C['t_blank'],C['frames']):.1f}% "
          f"nonblank {pct(C['t_nonblank'],C['frames']):.1f}%")
    print(f"over-firing frames (teacher blank & student non-blank): {C['overfire']} "
          f"= {pct(C['overfire'],C['t_blank']):.2f}% of teacher-blank frames")
    ov = max(C["overfire"], 1)
    print(f"  ├ shoulder (dist<={args.shoulder} of a teacher spike): {C['shoulder']} ({pct(C['shoulder'],ov):.1f}% of over-fire)")
    print(f"  │    ├ same token as that spike : {C['shoulder_same']} ({pct(C['shoulder_same'],ov):.1f}%)  ← clean span-spread")
    print(f"  │    └ different token          : {C['shoulder_diff']} ({pct(C['shoulder_diff'],ov):.1f}%)")
    print(f"  └ isolated (no spike within {args.shoulder}): {C['isolated']} ({pct(C['isolated'],ov):.1f}% of over-fire)  ← potential hallucination")
    print("  distance histogram (frames→count):",
          {k: dist_hist[k] for k in sorted(dist_hist)})


if __name__ == "__main__":
    main()
