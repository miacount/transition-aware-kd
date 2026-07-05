#!/usr/bin/env python3
"""Fine-tune a copy of the BPE teacher to be less peaky, via blank-only label priors.

Motivation. The post-hoc delta de-peak (build_span_kd_targets.py --teacher-only
--teacher-blank-penalty) suppresses blank AFTER the fact on a frozen teacher, so
occupancy can only widen ~2 frames before dark knowledge drifts. Here we instead
bake the de-peaking into the WEIGHTS: fine-tune the teacher with the label-prior
CTC loss (Huang et al., ICASSP 2024), which pushes probability off blank onto the
non-blank tokens during training. The model LEARNS a coherent, data-dependent
spread with no post-hoc ceiling, and label priors are also known to reduce CTC
emission delay.

Loss (blank penalty, in-weights analog of the post-hoc delta de-peak):
    adj_t(blank) = log p_t(blank) - delta     # delta > 0, nats, ramped
    adj_t(k)     = log p_t(k)                for k != blank
    L = CTC(adj, transcript)
Penalising blank makes blank-heavy alignments score worse, so training moves
probability onto non-blank tokens over more frames -> the learned posterior
de-peaks. Non-blank scores are untouched, so no rare-BPE path explosion.

NB: this is the OPPOSITE sign of a blank label-prior term (log_p[blank] -
alpha*log P(blank)); with log P(blank) < 0 that BOOSTS blank and never de-peaks
(observed empirically: P(blank) and token duration stayed flat).

We do NOT select by val CTC loss: de-peaking intentionally raises CTC loss a bit,
so best-val would pick the least-de-peaked model. We keep every epoch and the
final .nemo, and let downstream student WER decide. Per epoch we report val CTC
loss (accuracy proxy) and P(blank) / mean token duration (peakedness proxy) so
the accuracy-vs-de-peak tradeoff of each delta is visible.

Example:
  python scripts/finetune_teacher_depeak.py \
    --manifest data/train_clean_100.json --val-manifest data/dev_clean.json \
    --save-dir nemo_experiments/teacher_depeak_d6 \
    --blank-penalty 6.0 --epochs 8 --lr 1e-4 --batch 16
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from visualize_ctc_forward_backward import ctc_forward_backward  # noqa: E402


def ctc_blank_penalty_loss(log_probs, targets, input_lengths, target_lengths, delta, blank):
    """CTC loss with a POSITIVE blank penalty (in nats) subtracted from the blank
    score at every frame:  adj_t(blank) = log p_t(blank) - delta ;  adj_t(k)=log p_t(k).

    This is the in-weights analog of the post-hoc delta de-peak that we verified
    works. Penalising blank makes blank-heavy alignments score worse, so to
    minimise CTC loss the model must move probability onto non-blank tokens over
    more frames -> the learned posterior de-peaks. Non-blank scores are untouched,
    so no rare-BPE path explosion. delta=0 -> plain CTC.

    NOTE: this is the OPPOSITE sign of the old blank label-prior term
    (log_p[blank] - alpha*log P(blank)), which with log P(blank)<0 actually
    *boosted* blank and never de-peaked.
    """
    if delta == 0.0:
        adj = log_probs
    else:
        adj = log_probs.clone()
        adj[:, :, blank] = log_probs[:, :, blank] - delta
    return F.ctc_loss(
        adj.permute(1, 0, 2), targets, input_lengths, target_lengths,
        blank=blank, reduction="mean", zero_infinity=True,
    )


class WavDataset(Dataset):
    """Returns raw 16kHz waveform + BPE token ids (teacher does mel internally)."""
    def __init__(self, manifest, tokenizer, sample_rate, max_duration=18.0):
        self.rows = []
        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("duration", 0) <= max_duration:
                    self.rows.append(row)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != self.sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        toks = torch.tensor(self.tokenizer.text_to_ids(row["text"]), dtype=torch.long)
        return torch.from_numpy(wav).float(), toks


def collate(batch):
    wavs, toks = zip(*batch)
    B = len(wavs)
    wl = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    tl = torch.tensor([t.shape[0] for t in toks], dtype=torch.long)
    W = int(wl.max()); N = int(tl.max())
    wav_pad = torch.zeros(B, W)
    tok_pad = torch.zeros(B, N, dtype=torch.long)
    for i, (w, t) in enumerate(zip(wavs, toks)):
        wav_pad[i, :w.shape[0]] = w
        tok_pad[i, :t.shape[0]] = t
    return wav_pad, wl, tok_pad, tl


@torch.no_grad()
def mean_token_duration(model, wavs, wav_lens, toks, tok_lens, blank_id, n=4):
    """Participation-ratio token duration on a few utts = peakedness probe."""
    log_probs, enc_len, _ = model.forward(input_signal=wavs, input_signal_length=wav_lens)
    durs = []
    for b in range(min(n, wavs.shape[0])):
        T = int(enc_len[b].item()); L = int(tok_lens[b].item())
        if T <= 0 or L <= 0:
            continue
        lp = log_probs[b, :T].float().cpu().numpy()
        ids = toks[b, :L].cpu().numpy().tolist()
        tg = ctc_forward_backward(lp, ids, blank_id)["token_gamma"]  # (T, L)
        s1 = tg.sum(0); s2 = (tg ** 2).sum(0)
        eff = (s1 ** 2) / np.maximum(s2, 1e-12)
        v = s1 > 1e-6
        if v.any():
            durs.append(float(eff[v].mean()))
    return float(np.mean(durs)) if durs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--manifest", default="data/train_clean_100.json")
    ap.add_argument("--val-manifest", default="data/dev_clean.json")
    ap.add_argument("--tokenizer", default="tokenizer_1024")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--blank-penalty", type=float, default=6.0,
                    help="nats subtracted from blank score during CTC (de-peaking amount). "
                         "Analog of the post-hoc delta; sweep this (e.g. 3, 6, 9).")
    ap.add_argument("--penalty-warmup", type=int, default=2,
                    help="epochs to ramp penalty 0 -> --blank-penalty (epoch 1 is always 0)")
    ap.add_argument("--max-duration", type=float, default=18.0)
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="epochs to keep encoder frozen (decoder-only warm start)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    import nemo.collections.asr as nemo_asr
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher).to(args.device)
    blank_id = int(model.decoder.num_classes_with_blank - 1)
    vocab = int(model.decoder.num_classes_with_blank)
    sample_rate = int(model.cfg.preprocessor.sample_rate)
    tokenizer = SentencePieceTokenizer(model_path=os.path.join(args.tokenizer, "tokenizer.model"))
    print(f"[model] teacher loaded | vocab={vocab} blank_id={blank_id} sr={sample_rate}")

    train_dl = DataLoader(
        WavDataset(args.manifest, tokenizer, sample_rate, args.max_duration),
        batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=6, pin_memory=True, drop_last=True)
    val_dl = DataLoader(
        WavDataset(args.val_manifest, tokenizer, sample_rate, args.max_duration),
        batch_size=args.batch, shuffle=False, collate_fn=collate,
        num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        delta_eff = 0.0 if epoch == 1 else args.blank_penalty * min(1.0, (epoch - 1) / max(1, args.penalty_warmup))
        freeze_enc = epoch <= args.freeze_epochs
        for p in model.encoder.parameters():
            p.requires_grad = not freeze_enc

        model.train()
        # keep spec augment on for regularization; it's the teacher's own config
        blank_acc = 0.0
        frames = 0
        run = 0.0
        steps = 0
        for wavs, wl, toks, tl in train_dl:
            wavs, wl, toks, tl = wavs.to(args.device), wl.to(args.device), toks.to(args.device), tl.to(args.device)
            log_probs, enc_len, _ = model.forward(input_signal=wavs, input_signal_length=wl)
            loss = ctc_blank_penalty_loss(log_probs, toks, enc_len, tl, delta_eff, blank_id)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                pb = log_probs.exp()[:, :, blank_id]           # (B, T)
                mask = torch.arange(pb.shape[1], device=args.device)[None, :] < enc_len[:, None]
                blank_acc += float((pb * mask).sum())
                frames += int(mask.sum())
            run += loss.item(); steps += 1
            if steps % 200 == 0:
                print(f"  ep{epoch} step{steps} loss={run/steps:.3f}", flush=True)

        p_blank = blank_acc / max(frames, 1)

        # validation: clean CTC loss (delta=0) + peakedness probe
        model.eval()
        vloss = 0.0; nval = 0
        probe = None
        with torch.no_grad():
            for wavs, wl, toks, tl in val_dl:
                wavs, wl, toks, tl = wavs.to(args.device), wl.to(args.device), toks.to(args.device), tl.to(args.device)
                log_probs, enc_len, _ = model.forward(input_signal=wavs, input_signal_length=wl)
                vloss += ctc_blank_penalty_loss(log_probs, toks, enc_len, tl, 0.0, blank_id).item()
                nval += 1
                if probe is None:
                    probe = mean_token_duration(model, wavs, wl, toks, tl, blank_id)
        vloss /= max(nval, 1)
        print(f"Epoch {epoch:2d}/{args.epochs}  delta={delta_eff:.2f}  "
              f"train={run/steps:.3f}  val_ctc={vloss:.3f}  "
              f"P(blank)={p_blank:.3f}  eff_dur={probe:.2f}", flush=True)

        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "blank_penalty": args.blank_penalty, "delta_eff": delta_eff,
                    "args": vars(args)},
                   os.path.join(args.save_dir, f"epoch_{epoch:03d}.pt"))

    # final: save full NeMo model so build_span_kd_targets can restore_from it
    nemo_path = os.path.join(args.save_dir, "teacher_depeak.nemo")
    model.save_to(nemo_path)
    print(f"\nDone. final model: {nemo_path}")
    print("Use it as the WHERE source in build_span_kd_targets.py (keep original teacher for WHAT).")


if __name__ == "__main__":
    main()
