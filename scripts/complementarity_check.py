#!/usr/bin/env python
"""Error-complementarity gate for the span-KD width pyramid (Method 3).

Decodes TWO students (e.g. δ=3 and δ=6) greedily on the same dev sets, computes
per-utterance word edit distance for each, and reports:
  - each student's corpus WER (reproduction sanity vs eval)
  - ORACLE-COMBINE WER: per utterance take the hyp with fewer errors, recompute
    corpus WER. This is the upper bound of any combination of the two widths.
  - who-wins split + hypothesis disagreement rate.

Decision:  oracle ~= min(single)  -> widths redundant, pyramid has no room.
           oracle << min(single)  -> complementary error profiles, pyramid worth a probe.
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_student import load_model, make_split_cfgs  # noqa: E402


def greedy_ctc(log_probs, enc_len, blank_id):
    """log_probs (B,T,V) -> list of token-id lists (collapsed, blank-removed)."""
    argm = log_probs.argmax(-1)  # (B,T)
    outs = []
    for b in range(argm.shape[0]):
        T = int(enc_len[b].item())
        prev = -1
        seq = []
        for t in range(T):
            p = int(argm[b, t].item())
            if p != prev and p != blank_id:
                seq.append(p)
            prev = p
        outs.append(seq)
    return outs


def wer_edits(ref_words, hyp_words):
    """word-level Levenshtein edit count."""
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


@torch.no_grad()
def decode_split(model, ds_cfg, device, blank_id, tok):
    dl = model._make_dataloader_from_cfg(ds_cfg, shuffle=False)
    refs, hyps = [], []
    for batch in dl:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        log_probs, enc_len, _ = model.forward(input_signal=batch["wavs"],
                                              input_signal_length=batch["wav_lens"])
        seqs = greedy_ctc(log_probs, enc_len, blank_id)
        for b, seq in enumerate(seqs):
            hyps.append(tok.ids_to_text(seq))
            L = int(batch["token_lens"][b].item())
            ref_ids = batch["tokens"][b, :L].tolist()
            refs.append(tok.ids_to_text(ref_ids))
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--ckpt-a", required=True, help="student A (e.g. d3)")
    ap.add_argument("--ckpt-b", required=True, help="student B (e.g. d6)")
    ap.add_argument("--label-a", default="d3")
    ap.add_argument("--label-b", default="d6")
    ap.add_argument("--manifest", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"[load] A={args.label_a}: {args.ckpt_a}")
    mA, cfg = load_model(args.config, args.ckpt_a, device)
    print(f"[load] B={args.label_b}: {args.ckpt_b}")
    mB, _ = load_model(args.config, args.ckpt_b, device)
    blank_id = int(mA.decoder.num_classes_with_blank - 1)
    tok = mA.tokenizer

    G = {"tot_w": 0, "eA": 0, "eB": 0, "eO": 0,
         "A_better": 0, "B_better": 0, "tie": 0, "disagree": 0, "n": 0,
         "both_ok": 0, "A_ok": 0, "B_ok": 0}

    for ds in make_split_cfgs(cfg, args.manifest):
        name = ds.get("name")
        refsA, hypsA = decode_split(mA, ds, device, blank_id, tok)
        refsB, hypsB = decode_split(mB, ds, device, blank_id, tok)
        assert refsA == refsB, f"ref mismatch in {name} (dataloader order differs)"
        sw = seA = seB = seO = 0
        aB = bB = tie = dis = 0
        for ref, hA, hB in zip(refsA, hypsA, hypsB):
            rw = ref.split()
            wA, wB = hA.split(), hB.split()
            dA, dB = wer_edits(rw, wA), wer_edits(rw, wB)
            nw = max(len(rw), 1)
            sw += len(rw); seA += dA; seB += dB; seO += min(dA, dB)
            if dA < dB: aB += 1
            elif dB < dA: bB += 1
            else: tie += 1
            if hA != hB: dis += 1
            G["both_ok"] += int(dA == 0 and dB == 0)
            G["A_ok"] += int(dA == 0); G["B_ok"] += int(dB == 0)
        G["tot_w"] += sw; G["eA"] += seA; G["eB"] += seB; G["eO"] += seO
        G["A_better"] += aB; G["B_better"] += bB; G["tie"] += tie
        G["disagree"] += dis; G["n"] += len(refsA)
        print(f"\n[{name}]  {args.label_a} {100*seA/sw:.2f}  |  {args.label_b} {100*seB/sw:.2f}  "
              f"|  oracle {100*seO/sw:.2f}   (n={len(refsA)})")

    tw = G["tot_w"]
    wA, wB, wO = 100*G["eA"]/tw, 100*G["eB"]/tw, 100*G["eO"]/tw
    lo = min(wA, wB)
    print("\n" + "=" * 60)
    print(f"CORPUS WER   {args.label_a}={wA:.2f}   {args.label_b}={wB:.2f}   oracle-combine={wO:.2f}")
    print(f"oracle gain vs best single ({'A' if wA<wB else 'B'})  :  {lo-wO:+.2f} abs  "
          f"({100*(lo-wO)/lo:.1f}% rel)")
    print(f"per-utt  {args.label_a}_better={G['A_better']}  {args.label_b}_better={G['B_better']}  "
          f"tie={G['tie']}   (n={G['n']})")
    print(f"hyp disagreement rate : {100*G['disagree']/G['n']:.1f}%  "
          f"of utts the two students output different text")
    print(f"exact-correct utts    : {args.label_a}={G['A_ok']}  {args.label_b}={G['B_ok']}  "
          f"both={G['both_ok']}  either={G['A_ok']+G['B_ok']-G['both_ok']}")
    print("=" * 60)
    print("READ: oracle≈min(single) -> redundant, KILL pyramid. "
          "oracle≪min -> complementary, probe {3,6}.")


if __name__ == "__main__":
    main()
