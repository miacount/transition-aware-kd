#!/usr/bin/env python3
"""
Train a phoneme CTC soft aligner with full label-prior correction.

This follows the label-prior idea used for less-peaky CTC forced alignment:

    score_t(k) = log p_t(k) - alpha * log P(k)

where P(k) is the empirical frame-level label prior estimated from the aligner
posterior. Unlike scripts/train_soft_aligner.py, this applies the prior to the
whole phoneme vocabulary, which is much safer than doing it for 1024 BPE units.

Lexicon format:
    WORD PH1 PH2 PH3 ...

Example:
    python scripts/train_phoneme_soft_aligner.py \
      --manifest data/train_clean_100.json \
      --val-manifest data/dev_clean.json \
      --lexicon data/cmudict.dict \
      --save-dir nemo_experiments/phoneme_soft_aligner \
      --epochs 20 --alpha 0.3
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phoneme_utils import (  # noqa: E402
    load_lexicon,
    make_phone_vocab,
    save_phone_vocab,
    text_to_phone_ids,
)
from train_soft_aligner import SoftAligner  # noqa: E402


def _wav_to_mel(wav: np.ndarray, sr: int = 16000) -> torch.Tensor:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000,
        power=2.0,
    )
    mel = np.log(mel + 1e-6)
    mean = mel.mean(axis=1, keepdims=True)
    std = mel.std(axis=1, keepdims=True)
    mel = (mel - mean) / (std + 1e-5)
    return torch.from_numpy(mel.T).float()



def ctc_with_label_priors(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    log_priors,
    alpha,
    blank,
):
    """Paper-faithful CTC over prior-adjusted, unnormalized label scores.

    PyTorch CTCLoss expects normalized log-probabilities. Label-prior
    correction produces unnormalized path scores, so we compute the CTC forward
    recursion directly with torch.logsumexp. Gradients still flow to log_probs.
    """
    if alpha == 0.0:
        scores = log_probs
    else:
        scores = log_probs - alpha * log_priors.to(log_probs.device).view(1, 1, -1)
    return ctc_loss_from_scores(scores, targets, input_lengths, target_lengths, blank)


def ctc_loss_from_scores(scores, targets, input_lengths, target_lengths, blank):
    """CTC negative log-likelihood for unnormalized per-label scores.

    Args:
        scores: (B, T, V), log-score for each frame and label.
        targets: (B, L), padded target labels without blanks.
    """
    batch_size, max_frames, _ = scores.shape
    max_labels = targets.shape[1]
    max_states = 2 * max_labels + 1
    device = scores.device
    dtype = scores.dtype
    neg_inf = torch.finfo(dtype).min / 4

    ext = targets.new_full((batch_size, max_states), blank)
    ext[:, 1::2] = targets
    ext_lens = 2 * target_lengths + 1

    state_idx = torch.arange(max_states, device=device).unsqueeze(0)
    state_valid = state_idx < ext_lens.unsqueeze(1)
    label_scores = torch.gather(
        scores,
        2,
        ext.unsqueeze(1).expand(batch_size, max_frames, max_states),
    )

    is_label = ext != blank
    different_from_two_back = torch.zeros(batch_size, max_states, dtype=torch.bool, device=device)
    if max_states > 2:
        different_from_two_back[:, 2:] = ext[:, 2:] != ext[:, :-2]
    can_skip = is_label & different_from_two_back & state_valid

    alpha = scores.new_full((batch_size, max_states), neg_inf)
    alpha[:, 0] = label_scores[:, 0, 0]
    if max_states > 1:
        has_first_label = target_lengths > 0
        alpha[:, 1] = torch.where(
            has_first_label,
            label_scores[:, 0, 1],
            scores.new_full((batch_size,), neg_inf),
        )
    alpha = alpha.masked_fill(~state_valid, neg_inf)

    pad1 = scores.new_full((batch_size, 1), neg_inf)
    pad2 = scores.new_full((batch_size, 2), neg_inf)

    for t in range(1, max_frames):
        stay = alpha
        prev1 = torch.cat([pad1, alpha[:, :-1]], dim=1)
        prev2 = torch.cat([pad2, alpha[:, :-2]], dim=1)
        prev2 = torch.where(can_skip, prev2, torch.full_like(prev2, neg_inf))
        candidates = torch.stack([stay, prev1, prev2], dim=0)
        new_alpha = torch.logsumexp(candidates, dim=0) + label_scores[:, t]
        frame_valid = t < input_lengths
        alpha = torch.where(
            frame_valid.unsqueeze(1),
            new_alpha.masked_fill(~state_valid, neg_inf),
            alpha,
        )

    batch_idx = torch.arange(batch_size, device=device)
    last = (ext_lens - 1).clamp_min(0)
    prev = (ext_lens - 2).clamp_min(0)
    log_z = torch.logsumexp(
        torch.stack([alpha[batch_idx, last], alpha[batch_idx, prev]], dim=0),
        dim=0,
    )
    finite = torch.isfinite(log_z)
    if not finite.any():
        return -log_z.new_zeros(())
    losses = -log_z[finite] / target_lengths[finite].clamp_min(1).to(dtype)
    return losses.mean()


def clean_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank):
    return F.ctc_loss(
        log_probs.permute(1, 0, 2),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction='mean',
        zero_infinity=True,
    )


class PhonemeDataset(Dataset):
    def __init__(
        self,
        manifest,
        lexicon,
        phone_to_id,
        max_duration=20.0,
        unk_policy="skip",
    ):
        self.rows = []
        self.lexicon = lexicon
        self.phone_to_id = phone_to_id
        self.unk_policy = unk_policy
        skipped = 0
        missing_words = {}

        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("duration", 0) > max_duration:
                    continue
                ids, missing = text_to_phone_ids(
                    row["text"], lexicon, phone_to_id, unk_policy=unk_policy
                )
                if ids is None or len(ids) == 0:
                    skipped += 1
                    for w in missing:
                        missing_words[w] = missing_words.get(w, 0) + 1
                    continue
                row["_phone_ids"] = ids
                self.rows.append(row)

        if not self.rows:
            raise ValueError(
                f"no usable rows in {manifest}; provide a larger lexicon or use --unk-policy drop"
            )
        print(
            f"[data] {manifest}: kept={len(self.rows)} skipped={skipped} "
            f"vocab={len(phone_to_id)}"
        )
        if missing_words:
            top = sorted(missing_words.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print(f"[data] top missing words: {top}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        mel = _wav_to_mel(wav, sr)
        tokens = torch.tensor(row["_phone_ids"], dtype=torch.long)
        return mel, torch.tensor(mel.shape[0]), tokens


def collate_fn(batch):
    mels, mel_lens, tokens = zip(*batch)
    batch_size = len(mels)
    t_max = max(m.shape[0] for m in mels)
    n_max = max(t.shape[0] for t in tokens)
    mel_pad = torch.zeros(batch_size, t_max, mels[0].shape[1])
    tok_pad = torch.zeros(batch_size, n_max, dtype=torch.long)
    ml = torch.stack(mel_lens).long()
    tl = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    for i, (mel, tok) in enumerate(zip(mels, tokens)):
        mel_pad[i, : mel.shape[0]] = mel
        tok_pad[i, : tok.shape[0]] = tok
    return mel_pad, ml, tok_pad, tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train_clean_100.json")
    ap.add_argument("--val-manifest", default="data/dev_clean.json")
    ap.add_argument("--lexicon", default="", help="CMU-style WORD PH1 PH2 lexicon")
    ap.add_argument("--save-dir", default="nemo_experiments/phoneme_soft_aligner")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--alpha-warmup", type=int, default=5)
    ap.add_argument("--prior-floor", type=float, default=1e-5)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--unk-policy", choices=["skip", "drop"], default="skip")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    lexicon = load_lexicon(args.lexicon or None, strip_stress_marks=True)
    phone_to_id, id_to_phone, blank_id = make_phone_vocab(lexicon)
    vocab_size = blank_id + 1
    save_phone_vocab(os.path.join(args.save_dir, "phones.tsv"), id_to_phone)
    print(f"[phones] vocab={vocab_size} blank_id={blank_id}")

    model = SoftAligner(input_dim=80, hidden=args.hidden, vocab=vocab_size).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params")

    train_ds = PhonemeDataset(
        args.manifest,
        lexicon,
        phone_to_id,
        max_duration=args.max_duration,
        unk_policy=args.unk_policy,
    )
    val_ds = PhonemeDataset(
        args.val_manifest,
        lexicon,
        phone_to_id,
        max_duration=args.max_duration,
        unk_policy=args.unk_policy,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log_priors = torch.full((vocab_size,), -np.log(vocab_size), device=args.device)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
            alpha_eff = 0.0
        else:
            ramp = min(1.0, (epoch - 1) / max(1, args.alpha_warmup))
            alpha_eff = args.alpha * ramp

        model.train()
        prior_acc = torch.zeros(vocab_size, device=args.device)
        prior_frames = 0
        train_loss = 0.0
        n_steps = 0

        for mel, mel_lens, phones, phone_lens in train_dl:
            mel = mel.to(args.device)
            mel_lens = mel_lens.to(args.device)
            phones = phones.to(args.device)
            phone_lens = phone_lens.to(args.device)

            log_probs, out_lens = model(mel, mel_lens)
            loss = ctc_with_label_priors(
                log_probs,
                phones,
                out_lens,
                phone_lens,
                log_priors,
                alpha_eff,
                blank_id,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                probs = log_probs.exp()
                for b in range(len(out_lens)):
                    frames = int(out_lens[b].item())
                    prior_acc += probs[b, :frames].sum(dim=0)
                    prior_frames += frames

            train_loss += float(loss.detach().item())
            n_steps += 1
            if n_steps % 200 == 0:
                print(f"  ep{epoch} step{n_steps:4d} loss={train_loss/n_steps:.3f}", flush=True)

        priors = (prior_acc / max(prior_frames, 1)).clamp_min(args.prior_floor)
        priors = priors / priors.sum()
        log_priors = priors.log()
        p_blank = float(priors[blank_id].item())
        scheduler.step()

        model.eval()
        val_prior_loss = 0.0
        val_clean_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for mel, mel_lens, phones, phone_lens in val_dl:
                mel = mel.to(args.device)
                mel_lens = mel_lens.to(args.device)
                phones = phones.to(args.device)
                phone_lens = phone_lens.to(args.device)
                log_probs, out_lens = model(mel, mel_lens)
                loss_prior = ctc_with_label_priors(
                    log_probs,
                    phones,
                    out_lens,
                    phone_lens,
                    log_priors,
                    alpha_eff,
                    blank_id,
                )
                loss_clean = clean_ctc_loss(
                    log_probs,
                    phones,
                    out_lens,
                    phone_lens,
                    blank_id,
                )
                val_prior_loss += float(loss_prior.item())
                val_clean_loss += float(loss_clean.item())
                n_val += 1
        val_prior_loss /= max(n_val, 1)
        val_clean_loss /= max(n_val, 1)
        train_avg = train_loss / max(n_steps, 1)
        print(
            f"Epoch {epoch:3d}/{args.epochs} train={train_avg:.3f} "
            f"val_prior={val_prior_loss:.3f} val_clean={val_clean_loss:.3f} "
            f"alpha_eff={alpha_eff:.3f} P(blank)={p_blank:.3f}"
        )

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "log_priors": log_priors.detach().cpu(),
            "phone_to_id": phone_to_id,
            "id_to_phone": id_to_phone,
            "blank_id": blank_id,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch:03d}.pt"))
        if val_prior_loss < best_val:
            best_val = val_prior_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"  -> best checkpoint saved (val_prior={best_val:.3f})")

    print(f"Done. Best val_prior={best_val:.3f}")


if __name__ == "__main__":
    main()
