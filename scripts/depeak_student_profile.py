#!/usr/bin/env python
"""Reproduce the METHOD_REPORT §6 de-peak table for arbitrary student ckpts.

Per model, over N test-clean utterances:
  blank%          fraction of frames whose greedy argmax is blank
  frames/token    non-blank frames / number of collapsed greedy tokens
  emit            mean p(argmax) on non-blank frames  (per-frame confidence)
  spike offset    mean |t_S(u) - t_T(u)| over transcript tokens, where t(u) is
                  argmax_t gamma(t,u) from each model's own transcript-constrained
                  forward-backward (teacher frame grid; student resampled if rates differ)
"""
import argparse, os, sys, json
import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/workspace/transition_aware_kd"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate_student import load_model            # noqa: E402
from ctc_fb import batched_ctc_token_gamma         # noqa: E402


def read_manifest(p, limit):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_audio(path, sr_target=16000):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == sr_target, f"{sr} != {sr_target}"
    return wav


@torch.no_grad()
def posterior(model, wav, device):
    wt = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    wl = torch.tensor([wt.shape[1]], dtype=torch.long, device=device)
    lp, el, _ = model.forward(input_signal=wt, input_signal_length=wl)
    return lp[0, :int(el[0])].float()


@torch.no_grad()
def spikes(lp, ids, blank):
    """argmax_t gamma(t,u) per transcript token, from this model's own FB."""
    T = lp.shape[0]
    g = batched_ctc_token_gamma(
        lp.unsqueeze(0), torch.tensor([T], device=lp.device),
        torch.tensor([ids], device=lp.device),
        torch.tensor([len(ids)], device=lp.device), blank)
    return g[0].argmax(dim=1).cpu().numpy(), T   # (N,)


def greedy_stats(lp, blank):
    p, am = lp.exp().max(dim=1)[0].cpu().numpy(), lp.argmax(dim=1).cpu().numpy()
    T = len(am)
    nb = am != blank
    # collapsed token count (CTC greedy decode)
    prev, ntok = -1, 0
    for a in am:
        if a != blank and a != prev:
            ntok += 1
        prev = a
    return dict(T=T, blank=int((~nb).sum()), nb=int(nb.sum()), ntok=ntok,
                emit_sum=float(p[nb].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/test_clean.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/student_base.yaml"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "tokenizer_1024"))
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--ckpt", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device(args.device)
    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(dev).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    tok = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    models = {"teacher": teacher}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        m, _ = load_model(args.config, path, dev)
        models[name] = m

    rows = read_manifest(args.manifest, args.limit)
    acc = {k: dict(T=0, blank=0, nb=0, ntok=0, emit_sum=0.0, off=[], n=0) for k in models}

    for i, r in enumerate(rows):
        wav = load_audio(r["audio_filepath"])
        ids = [int(x) for x in tok.text_to_ids(r["text"]) if int(x) != blank]
        if not ids:
            continue
        lps, sp = {}, {}
        for k, m in models.items():
            lp = posterior(m, wav, dev)
            lps[k] = lp
            g = greedy_stats(lp, blank)
            for f in ("T", "blank", "nb", "ntok"):
                acc[k][f] += g[f]
            acc[k]["emit_sum"] += g["emit_sum"]
            sp[k], _ = spikes(lp, ids, blank)
        Tt = lps["teacher"].shape[0]
        for k in models:
            if k == "teacher":
                continue
            Ts = lps[k].shape[0]
            s = sp[k].astype(np.float64) * (Tt / max(Ts, 1))   # student grid -> teacher grid
            acc[k]["off"].extend(np.abs(s - sp["teacher"]).tolist())
            acc[k]["n"] += len(ids)
        if (i + 1) % 50 == 0:
            print(f"  ..{i+1}/{len(rows)}", flush=True)

    print(f"\nmanifest={args.manifest}  utts={len(rows)}\n")
    hdr = f"{'model':<22}{'blank%':>9}{'frames/token':>14}{'emit':>9}{'|spike off|':>13}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for k, a in acc.items():
        blankp = 100.0 * a["blank"] / max(a["T"], 1)
        fpt = a["nb"] / max(a["ntok"], 1)
        emit = a["emit_sum"] / max(a["nb"], 1)
        off = float(np.mean(a["off"])) if a["off"] else float("nan")
        out[k] = dict(blank_pct=blankp, frames_per_token=fpt, emit=emit, spike_off=off)
        print(f"{k:<22}{blankp:>9.1f}{fpt:>14.2f}{emit:>9.3f}{off:>13.2f}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
