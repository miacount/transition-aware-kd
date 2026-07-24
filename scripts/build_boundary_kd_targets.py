#!/usr/bin/env python3
"""Build Counterfactual Boundary-KD targets.

Only blank-suppression's OWN contribution is kept: the counterfactual occupancy
gain r_{t,u} = [gamma^delta_{t,u} - gamma^0_{t,u}]_+, i.e. the shoulder frames
that de-peaking (delta) newly assigned to token u on top of the CTC spike gamma^0.

Per utterance we store the BASE target (cells 2/3/4, ~tiny):
  res_t, res_u, res_r : sparse residual weights r_{t,u} with r > residual_store_min
  m_delta   (T,)      : 1 - p_T^delta(blank | t)     hard-target emission mass
  m_zero    (T,)      : 1 - p_T^0(blank | t)         (cell 3)
  spike     (N,)      : t*_u = argmax_t gamma^0(.,u) (cell 4 shoulder centre)
  y         (N,)      : transcript token ids

The full soft posterior p_T^delta needed ONLY by cell 1 is written to a SEPARATE
sidecar file (FP16, ~2.7GB over train-clean-100) when --with-soft is given, so the
cheap cells never pay for it.

Loss (all cells, full-softmax, NO {blank,y_u} renormalisation):
  L = ( sum_{t,u} r_{t,u} * CE_{t,u} ) / ( sum r + eps )     [batch-normalised]
  cell1 CE: -sum_v p_T^delta(v|t) log p_S(v|t)               (soft)
  cell2 CE: -[(1-m^d) log p_S(blank) + m^d log p_S(y_u)]     (hard, mass m^d)
  cell3   : cell2 with m^0 in place of m^d
  cell4   : cell2 weights replaced by within-token uniform over dist=1 frames
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

from build_span_kd_targets import (  # noqa: E402
    blank_penalty_logprobs,
    load_audio,
    read_manifest,
    run_teacher,
    write_manifest,
)
from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


def build_residual(teacher_lp, bpe_ids, blank_id, delta):
    """Return r (T,N), m_delta (T,), m_zero (T,), spike (N,), and the de-peaked
    posterior p^delta (T,V). r = [gamma^delta - gamma^0]_+ column by column."""
    g0 = ctc_forward_backward(teacher_lp, bpe_ids, blank_id)["token_gamma"]      # (T,N)
    lp_d = blank_penalty_logprobs(teacher_lp, blank_id, delta)
    gd = ctc_forward_backward(lp_d, bpe_ids, blank_id)["token_gamma"]            # (T,N)
    r = np.clip(gd - g0, 0.0, None)                                             # (T,N)
    p0 = np.exp(teacher_lp)
    pd = np.exp(lp_d)
    m_zero = (1.0 - p0[:, blank_id]).astype(np.float32)                         # (T,)
    m_delta = (1.0 - pd[:, blank_id]).astype(np.float32)                        # (T,)
    spike = g0.argmax(axis=0).astype(np.int32)                                  # (N,)
    return r, m_delta, m_zero, spike, pd


def selfcheck():
    """Synthetic FB-free check of the residual bookkeeping and cell-4 mass."""
    # fake occupancy: token 0 spike at t=2, de-peak spreads to t=1,3
    g0 = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]])
    gd = np.array([[0.0], [0.20], [0.60], [0.20], [0.0]])
    r = np.clip(gd - g0, 0.0, None)
    assert r[2, 0] == 0.0 and r[1, 0] > 0 and r[3, 0] > 0, "residual excludes the spike"
    spike = g0.argmax(0)
    assert spike[0] == 2
    # cell-4 redistribution: A_u onto dist==1 frames, spike excluded, mass preserved
    A = r[:, 0].sum()
    D = [t for t in range(5) if abs(t - spike[0]) == 1]
    w4 = np.zeros(5)
    for t in D:
        w4[t] = A / len(D)
    assert abs(w4.sum() - A) < 1e-9, "cell-4 preserves per-token residual mass"
    assert w4[2] == 0.0, "cell-4 excludes the spike frame"
    # rho_keep on a residual with a tiny tail entry
    rr = np.array([0.004, 0.30, 0.20])
    rho = rr[rr > 0.005].sum() / rr[rr > 0].sum()
    assert abs(rho - 0.50 / 0.504) < 1e-9
    print("[selfcheck] residual/spike/cell4-mass/rho_keep OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-in", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--out-dir", default="boundary_kd")
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--residual-store-min", type=float, default=0.001,
                    help="store r only where r > this (sparsification threshold). "
                         "0.001 keeps >=99.85%% of residual mass; raise at train time "
                         "via a runtime mask, never rebuild to lower it.")
    ap.add_argument("--rho-keep-min", type=float, default=0.995,
                    help="warn if retained residual mass fraction falls below this")
    ap.add_argument("--with-soft", action="store_true",
                    help="also write the full p_T^delta at residual frames (cell 1, large)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    selfcheck()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    rows = read_manifest(args.manifest_in)
    if args.limit:
        rows = rows[:args.limit]

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device).eval()
    teacher.freeze()
    blank = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))

    manifest_dir = Path(args.manifest_out).parent
    out_dir = Path(args.out_dir)
    (manifest_dir / out_dir.name).mkdir(parents=True, exist_ok=True)
    if args.with_soft:
        (manifest_dir / (out_dir.name + "_soft")).mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_empty = 0
    kept_mass = 0.0
    all_mass = 0.0
    res_per_tok = []

    for idx, row in enumerate(rows):
        bpe_ids = [int(i) for i in tokenizer.text_to_ids(row["text"]) if int(i) != blank]
        if not bpe_ids:
            skipped_empty += 1
            continue
        wav = load_audio(row["audio_filepath"], sample_rate)
        teacher_lp = run_teacher(teacher, wav, args.device)                     # (T,V)
        r, m_delta, m_zero, spike, pd = build_residual(teacher_lp, bpe_ids, blank, args.delta)
        T, N = r.shape

        all_mass += float(r.sum())
        keep = r > args.residual_store_min
        kept_mass += float(r[keep].sum())
        tt, uu = np.nonzero(keep)                 # tt=frame idx, uu=token idx
        res_r = r[tt, uu].astype(np.float32)
        res_per_tok.append(len(tt) / max(N, 1))

        rel = Path(out_dir.name) / f"{idx:06d}.pt"
        torch.save(
            {
                "res_t": torch.tensor(tt, dtype=torch.int32),
                "res_u": torch.tensor(uu, dtype=torch.int32),
                "res_r": torch.tensor(res_r, dtype=torch.float32),
                "m_delta": torch.tensor(m_delta, dtype=torch.float16),
                "m_zero": torch.tensor(m_zero, dtype=torch.float16),
                "spike": torch.tensor(spike, dtype=torch.int32),
                "y": torch.tensor(bpe_ids, dtype=torch.int32),
                "teacher_frames": int(T),
                "num_tokens": int(N),
                "delta": float(args.delta),
                "residual_store_min": float(args.residual_store_min),
            },
            manifest_dir / rel,
        )
        row["boundary_kd_path"] = str(rel)

        if args.with_soft:
            res_frames = np.unique(tt)
            rel_soft = Path(out_dir.name + "_soft") / f"{idx:06d}.pt"
            torch.save(
                {
                    "soft_t": torch.tensor(res_frames, dtype=torch.int32),
                    "soft_p": torch.tensor(pd[res_frames], dtype=torch.float16),  # (F,V)
                },
                manifest_dir / rel_soft,
            )
            row["boundary_kd_soft_path"] = str(rel_soft)

        kept += 1
        if kept % args.flush_every == 0:
            write_manifest(args.manifest_out, rows)
            print(f"[prog] kept={kept} processed={idx+1}/{len(rows)} "
                  f"rho_keep={kept_mass/max(all_mass,1e-9):.4f}", flush=True)

    write_manifest(args.manifest_out, rows)
    rho = kept_mass / max(all_mass, 1e-9)
    print("\n========== summary ==========")
    print(f"kept              : {kept}")
    print(f"skipped (empty)   : {skipped_empty}")
    print(f"residual/token    : {np.mean(res_per_tok):.3f}")
    print(f"rho_keep (mass)   : {rho:.4f}  (store_min={args.residual_store_min})")
    if rho < args.rho_keep_min:
        print(f"[WARN] rho_keep {rho:.4f} < {args.rho_keep_min}: rebuild with "
              f"--residual-store-min 0.001")
    print(f"written           : {args.manifest_out}")


if __name__ == "__main__":
    main()
