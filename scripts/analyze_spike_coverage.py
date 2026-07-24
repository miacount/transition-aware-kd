#!/usr/bin/env python3
"""Does the de-peaked occupancy actually cover where the STUDENT puts its spikes?

P2 (spike misalignment) says teacher and student fire the same token on different
frames, so a frame-to-frame KD comparison lands on the wrong frame. This script
measures that directly and asks whether the delta-widened occupancy is wide enough
to absorb it.

Per token u (transcript-constrained FB on each model's own posterior):
    t_T(u) = argmax_t  gamma_teacher(t, u)        teacher spike
    t_S(u) = argmax_t  gamma_student(t, u)        student spike
    offset = t_S - t_T

Coverage of the student's spike by the teacher's target support, per delta:
    cov_d(u)  = gamma_d_teacher(t_S(u), u)        occupancy mass sitting on the
                                                  frame the student actually used
    hit_d(u)  = 1[ cov_d(u) > eps ]

delta=0 is frame-KD's implicit support, so cov_0 vs cov_6 is exactly "how much of
the misalignment does the widening buy back". Stratified by |offset|.

Outputs: printed tables + summary json. The per-utterance illustration of the
same effect is scripts/fig2_atd_misalignment.py.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze_mfa_occupancy import (  # noqa: E402
    blank_penalty_logprobs,
    fb_token_gamma_fast,
    normalize_cols_occ,
    read_manifest,
    run_teacher,
)




def load_student(path, device):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecCTCModelBPE.restore_from(path, map_location=device)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_clean.json")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--students", nargs="+", default=[
        "no-KD=nemo_experiments/student-no-kd/2026-06-24_04-38-47/checkpoints/student-no-kd.nemo",
        "ATD (δ=6, w=20)=nemo_experiments/span-kd-teacher-d6-w20.0/2026-07-02_14-03-36/"
        "checkpoints/span-kd-teacher-d6-w20.0.nemo",
    ], help="NAME=path.nemo")
    ap.add_argument("--deltas", default="0,3,6,9,12")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.01, help="hit threshold on occupancy")
    ap.add_argument("--out", default="analysis/spike_coverage")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    assert deltas[0] == 0.0, "first delta must be 0 (frame-KD reference)"
    students = [s.rsplit("=", 1) for s in args.students]   # names may contain '='

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest, args.limit)
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sr = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    models = {name: load_student(p, args.device) for name, p in students}
    for m in models.values():
        m.freeze()

    acc = {name: {"off": [], "cov": {d: [] for d in deltas}} for name in models}

    for row in rows:
        utt = Path(row["audio_filepath"]).stem
        ids = [int(x) for x in tokenizer.text_to_ids(row["text"]) if int(x) != blank]
        if not ids:
            continue
        wav, file_sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert file_sr == sr

        t_lp = run_teacher(teacher, wav, args.device)
        T = t_lp.shape[0]
        occ = {d: normalize_cols_occ(fb_token_gamma_fast(
            blank_penalty_logprobs(t_lp, blank, d), ids, blank)) for d in deltas}
        t_spike = occ[0.0].argmax(axis=0)

        s_lp, s_spike = {}, {}
        for name, m in models.items():
            lp = run_teacher(m, wav, args.device)
            if lp.shape[0] != T:                      # different subsampling: skip
                continue
            g = normalize_cols_occ(fb_token_gamma_fast(lp, ids, blank))
            sp = g.argmax(axis=0)
            s_lp[name], s_spike[name] = lp, sp
            acc[name]["off"].append(sp - t_spike)
            for d in deltas:
                acc[name]["cov"][d].append(occ[d][sp, np.arange(len(ids))])


    # ----------------------------------------------------------------- tables --
    summary = {"eps": args.eps, "deltas": deltas, "students": {}}
    print(f"\n{'='*78}\nteacher-student spike misalignment and occupancy coverage"
          f"\n{'='*78}")
    for name in models:
        off = np.concatenate(acc[name]["off"])
        cov = {d: np.concatenate(acc[name]["cov"][d]) for d in deltas}
        aoff = np.abs(off)
        S = {"n_tokens": int(off.size),
             "mean_abs_offset": float(aoff.mean()),
             "median_abs_offset": float(np.median(aoff)),
             "mean_signed_offset": float(off.mean()),
             "frac_offset_0": float((aoff == 0).mean()),
             "frac_offset_le1": float((aoff <= 1).mean()),
             "by_stratum": {}}
        print(f"\n--- {name} ---")
        print(f"tokens={off.size}  mean|offset|={aoff.mean():.2f}  "
              f"median|offset|={np.median(aoff):.0f}  signed mean={off.mean():+.2f}  "
              f"exact hit={100*(aoff == 0).mean():.1f}%")
        hdr = f"{'|offset|':>9} {'tokens':>8} {'share':>7} " + \
              "".join(f"{'δ='+f'{d:g}':>16}" for d in deltas)
        print(hdr)
        print(f"{'':>9} {'':>8} {'':>7} " + "".join(f"{'mass / hit%':>16}" for d in deltas))
        strata = [("0", aoff == 0), ("1", aoff == 1), ("2", aoff == 2),
                  ("3", aoff == 3), (">=4", aoff >= 4), ("all", np.ones_like(aoff, bool))]
        for lab, m in strata:
            if m.sum() == 0:
                continue
            cells, rec = "", {}
            for d in deltas:
                mass = float(cov[d][m].mean())
                hit = float((cov[d][m] > args.eps).mean())
                rec[f"delta={d:g}"] = {"mass": mass, "hit": hit}
                cells += f"{mass:>8.3f} /{100*hit:>5.1f}"
            S["by_stratum"][lab] = {"n": int(m.sum()),
                                    "share": float(m.mean()), **rec}
            print(f"{lab:>9} {int(m.sum()):>8} {m.mean():>7.3f} {cells}")
        summary["students"][name] = S

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out / "offsets.npz",
                        **{f"{n}_off": np.concatenate(acc[n]["off"]) for n in models},
                        **{f"{n}_cov{d:g}": np.concatenate(acc[n]["cov"][d])
                           for n in models for d in deltas})

    print(f"\nsaved: {out}/summary.json, offsets.npz")


if __name__ == "__main__":
    main()
